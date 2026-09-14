# Integrating an External Costmap Plugin into Nav2

A step-by-step checklist for adding a **new** Nav2 costmap layer plugin to
an **already-working** Nav2 stack (this workspace's, or any other). Every
step here is the exact pattern already proven working by the three
plugins in this directory (`stac_costmap_plugin`,
`nav2_hazard_costmap_plugin`, `social_costmap_layer`) — copy one of them as
your starting point rather than writing from scratch.

A Nav2 costmap plugin is a C++ class that inherits `nav2_costmap_2d::Layer`
and gets loaded into `global_costmap` and/or `local_costmap` at runtime via
`pluginlib`. Nothing about this requires touching Nav2's own source code —
it's a separate package that Nav2 discovers at startup.

---

## 1. Package structure

```
my_costmap_plugin/
├── CMakeLists.txt
├── package.xml
├── my_plugin.xml                      # pluginlib class description (name it anything)
├── include/
│   └── my_costmap_plugin/
│       └── my_layer.hpp
├── src/
│   └── my_layer.cpp
├── config/                            # optional: example/default params
│   └── my_layer_params.yaml
└── scripts/                           # optional: a Python node feeding this layer
    └── my_input_publisher.py
```

Only `CMakeLists.txt`, `package.xml`, the plugin description XML, and the
`.hpp`/`.cpp` pair are required. `config/` and `scripts/` are there if your
layer needs an external data source (most social/dynamic layers do).

---

## How a plugin actually gets loaded (no launch step of its own)

A plugin package is never itself launched, started, or run — it has no
node, no PID, no entry in `ros2 node list`. It's a `.so` file that gets
pulled into an *already-running* process (`controller_server` for
`local_costmap`, the global costmap node, etc.) by `pluginlib`, at the
moment that process reads your `plugins:` list. Mechanically, in order:

1. **At build/install time**, `pluginlib_export_plugin_description_file(nav2_costmap_2d my_plugin.xml)`
   in your `CMakeLists.txt` writes one small marker file into your
   package's install tree:
   `share/ament_index/resource_index/nav2_costmap_2d__pluginlib__plugin/my_costmap_plugin`
   — its contents are just "here's where `my_plugin.xml` lives." This
   happens automatically as part of `colcon build`; nothing to run by hand.
   You can see it for real, on an already-built package, without launching
   anything:
   ```bash
   cat $(ros2 pkg prefix stac_costmap_plugin)/share/ament_index/resource_index/nav2_costmap_2d__pluginlib__plugin/stac_costmap_plugin
   ```
2. **At runtime**, when `controller_server`/the costmap node constructs a
   `pluginlib::ClassLoader<nav2_costmap_2d::Layer>("nav2_costmap_2d")` (Nav2
   does this once, internally, on startup — not something you write), that
   constructor scans *every* package on `AMENT_PREFIX_PATH` (i.e. every
   workspace you've `source install/setup.bash`'d) for marker files of
   exactly that resource type, and reads all the plugin XMLs it finds. This
   is why your terminal needs to have sourced the workspace containing your
   plugin *before* Nav2 starts — otherwise it's invisible to this scan even
   though the `.so` is built and sitting on disk.
3. From that scan, pluginlib now has an in-memory map from every
   `<class type="...">` string to which library file provides it. When your
   yaml's `plugins:` list names `my_costmap_plugin::MyLayer`, the costmap
   node calls `class_loader->createSharedInstance("my_costmap_plugin::MyLayer")`
   — this is a plain `dlopen()` of `libmy_costmap_plugin.so`, **inside the
   costmap node's own process**, followed by calling your class's
   constructor. Your plugin is now live, running inside a process that was
   already running before it existed.

The practical upshot: rebuilding your plugin package is enough to change
its behavior on the *next* Nav2 restart — there's no separate "deploy" or
"register" step. But because discovery happens once, at that host
process's startup, adding/removing a plugin from the `plugins:` list, or
rebuilding its `.so`, only takes effect after that host process (usually
the whole Nav2 bringup) restarts — not while it's running.

---

## 2. Write the plugin class

`include/my_costmap_plugin/my_layer.hpp` — the minimum a `Layer` needs:

```cpp
#pragma once
#include "nav2_costmap_2d/layer.hpp"
#include "rclcpp/rclcpp.hpp"

namespace my_costmap_plugin
{
class MyLayer : public nav2_costmap_2d::Layer
{
public:
  MyLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override {}
  bool isClearable() override {return false;}
};
}  // namespace my_costmap_plugin
```

`src/my_layer.cpp` — the four things every plugin must do:

```cpp
#include "my_costmap_plugin/my_layer.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace my_costmap_plugin
{

void MyLayer::onInitialize()
{
  // 1. Declare + read parameters, scoped by this layer's instance name.
  //    (name_ is set by Nav2 to whatever key you give it in the yaml —
  //    see Section 4. Never hardcode a param namespace.)
  declareParameter("enabled", rclcpp::ParameterValue(true));
  auto node = node_.lock();
  node->get_parameter(name_ + ".enabled", enabled_);

  // 2. Subscribe to whatever external input this layer needs (a topic,
  //    a TF lookup, nothing at all for a static layer).
  current_ = true;
}

void MyLayer::updateBounds(
  double, double, double, double * min_x, double * min_y,
  double * max_x, double * max_y)
{
  // Expand the dirty window to cover whatever you're about to paint.
  // If you don't touch min/max here, updateCosts below never gets called
  // for that area.
}

void MyLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j,
  int max_i, int max_j)
{
  // Write costs (0-254) into master_grid within [min_i,max_i)x[min_j,max_j).
  // Always max() with the existing cost unless you intend to erase what
  // layers below you already painted.
}

}  // namespace my_costmap_plugin

// 3. This one line is what makes pluginlib able to instantiate the class.
PLUGINLIB_EXPORT_CLASS(my_costmap_plugin::MyLayer, nav2_costmap_2d::Layer)
```

## 3. Register the plugin (pluginlib)

`my_plugin.xml`:

```xml
<class_libraries>
  <library path="my_costmap_plugin">
    <class type="my_costmap_plugin::MyLayer" base_class_type="nav2_costmap_2d::Layer">
      <description>One-line description of what this layer does.</description>
    </class>
  </library>
</class_libraries>
```

`package.xml` — add inside `<export>`:

```xml
<export>
  <build_type>ament_cmake</build_type>
  <nav2_costmap_2d plugin="${prefix}/my_plugin.xml" />
</export>
```

`CMakeLists.txt` — the parts specific to a plugin (add these to the usual
`ament_cmake` boilerplate):

```cmake
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

add_library(my_costmap_plugin SHARED src/my_layer.cpp)
target_include_directories(my_costmap_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(my_costmap_plugin nav2_costmap_2d pluginlib rclcpp)

# The argument here (nav2_costmap_2d) is what Nav2 actually searches when
# loading Layer plugins — it must match the <export> tag name above.
pluginlib_export_plugin_description_file(nav2_costmap_2d my_plugin.xml)

install(TARGETS my_costmap_plugin
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/)
install(FILES my_plugin.xml DESTINATION share/${PROJECT_NAME})
```

Build it on its own first, before touching Nav2's config at all:

```bash
cd ~/tiago_public_ws
colcon build --packages-select my_costmap_plugin --symlink-install
source install/setup.bash
```

If this build succeeds and the package shows up in `ros2 pkg list | grep my_costmap_plugin`,
the plugin exists and is installed — it just isn't loaded by anything yet.

---

## 4. Wire it into the already-working Nav2 stack

This is the only part that touches your existing, working setup — find the
params yaml your `navigation_launch.py` / `bringup_launch.py` already
passes as `params_file` (in this workspace: `pmb2_navigation/pmb2_2dnav/config/nav_public_sim.yaml`).

Under `global_costmap` and/or `local_costmap` (pick whichever makes sense —
static/rarely-changing layers usually go in `global_costmap`, anything
robot-proximity/dynamic usually goes in `local_costmap`):

```yaml
global_costmap:
  global_costmap:
    ros__parameters:
      plugins: ["static_layer", "obstacle_layer", "my_layer", "inflation_layer"]
      #                                            ^^^^^^^^^ add your layer's key here
      my_layer:
        plugin: "my_costmap_plugin::MyLayer"   # matches <class type="..."> above
        enabled: true
        # ... your layer's own params
```

Two things that matter and are easy to get backwards:

- **Order matters.** Put your layer *before* `inflation_layer` in the list
  — inflation reads the master grid and spreads cost outward from
  whatever's already there, so it needs to run last to inflate your
  layer's cost too.
- **`name_`** inside the C++ class is set to whatever key you use here
  (`my_layer` in this example) — that's what `name_ + ".enabled"` in
  `onInitialize()` resolves to. Rename the yaml key and the param
  namespace renames with it, automatically.

Rebuild (if you changed the plugin), then **restart the whole Nav2
bringup** — costmap plugins are loaded once when `global_costmap`/
`local_costmap` start up, not hot-reloadable. Changing the `plugins:` list
always needs a relaunch; changing an already-loaded plugin's own
parameters may not, if that plugin implements a dynamic parameter callback
(see `nav2_hazard_costmap_plugin` for an example — an *existing* zone's
`zones.<name>.type/x/y/radius/severity` are all `ros2 param set`-able live;
adding a brand-new zone name still needs a relaunch).

---

## 5. Verify it actually loaded

```bash
# 1. Is the plugin in the running costmap's plugin list?
ros2 param get /global_costmap/global_costmap plugins

# 2. Did it initialize without error? Look for these two lines in the
#    controller_server / global_costmap startup log:
#      Using plugin "my_layer"
#      Initialized plugin "my_layer"

# 3. Is its input topic (if any) actually connected?
ros2 topic info /my_layer_input --verbose   # publisher AND subscriber counts > 0

# 4. Is it painting cost? Look at the costmap in RViz (Map display,
#    Color Scheme: costmap, topic /global_costmap/costmap or
#    /local_costmap/costmap), or sample it directly:
ros2 topic echo /global_costmap/costmap --once | grep -c ' 1[0-9][0-9]\| 2[0-9][0-9]'
```

If step 1 doesn't list your plugin: the yaml wasn't picked up — check you
restarted the right launch and are pointing at the params file you think
you are (`ros2 param get /global_costmap/global_costmap plugins` shows the
live truth, the yaml on disk doesn't).

If step 2 doesn't appear at all: pluginlib couldn't find or load the
class — almost always a `plugin:` string that doesn't match `<class
type="...">` in your `my_plugin.xml`, or the library isn't actually
installed (`ros2 pkg prefix my_costmap_plugin` then check
`lib/libmy_costmap_plugin.so` exists).

---

## Pitfalls that cost real debugging time (all hit while building this workspace's three plugins)

- **Relative topic names inside a costmap plugin resolve against the
  costmap's own node namespace, not the global namespace.** `local_costmap`
  runs its node as `/local_costmap`, so a plugin parameter like
  `topic: "my_input"` silently becomes `/local_costmap/my_input` — which
  your publisher, running as a plain top-level node, never talks to. If
  your layer's input comes from outside the costmap's own namespace
  (nearly always true), use a leading-slash **absolute** topic name:
  `topic: "/my_input"`.
- **A costmap plugin with no data yet must not crash or spam errors** —
  `updateBounds`/`updateCosts` get called every costmap cycle from the
  moment the node starts, likely before your input topic has published
  anything. Guard with an "have I received anything yet" flag and return
  early.
- **TF frame mismatches fail silently.** If your layer consumes data in a
  frame other than the costmap's global frame (`odom` for `local_costmap`,
  `map` for `global_costmap`), you need a `tf_->lookupTransform(...)` and
  it needs a frame that's actually being published. A frame that "sounds
  right" but isn't wired to anything (we shipped one defaulting to
  `"world"` with no such TF frame in this system) means `updateCosts` just
  never paints anything, with no error — check `ros2 run tf2_ros
  tf2_echo <global_frame> <your_frame>` actually returns a transform.
- **`package.xml`'s `<export>` tag name and `pluginlib_export_plugin_description_file`'s
  first argument should both be `nav2_costmap_2d`, consistently.** Mismatches
  between them have been observed to still work on some ROS 2 Humble
  installs (ament_index resource lookup is more forgiving than you'd
  expect), but don't rely on that — keep them matching so it's guaranteed
  portable.
- **`severity`/`amplitude`-style cost-scaling values should be clamped to keep the resulting cost in `[0, 254]`.**
  254 is `nav2_costmap_2d::LETHAL_OBSTACLE` — a plugin that paints 254
  makes cells genuinely impassable, not just expensive. Know which one you
  mean.

---

## Going from simulation to a real robot

Everything above is identical on real hardware — a Nav2 costmap plugin
doesn't know or care whether the master grid it's writing into came from a
simulated or a real robot's sensors. What changes:

- **`use_sim_time` must be `false`** (or simply not set — that's the
  default) in whatever params file you use on the robot. Leaving it `true`
  makes every node wait on `/clock`, which nothing publishes on real
  hardware, and the whole stack hangs.
- **Your input topic needs a real publisher.** In this workspace, `/people`
  and `/social_costmap` are fed by scripts that read Gazebo's ground-truth
  `/gazebo/model_states` — that topic doesn't exist on a real robot. You
  need an actual perception pipeline (a person/leg detector, a tracked-object
  publisher, whatever's appropriate) publishing the same message type on
  the same topic name. The costmap plugin itself needs zero changes for
  this — it only ever sees the topic, never Gazebo.
- **Test the input and the plugin separately before combining them on the
  robot.** Record or fake a few messages on the input topic (`ros2 topic
  pub`, or a rosbag) and confirm the plugin paints cost correctly with
  Nav2 running but the robot disabled/lifted, *before* trusting it to
  influence a moving robot's obstacle avoidance.
- **Watch real CPU, not just sim wall-clock.** `cutoff` (STAC),
  `resolution`/`size_x`/`size_y` (proxemics grid) and similar
  cost/efficiency knobs that seemed fine on a development workstation may
  not fit inside the real robot's `controller_frequency` /
  `update_frequency` budget on embedded compute — check with `ros2 topic
  hz` on the costmap output and `top`/`htop` on the robot's onboard PC
  under real load before deploying.
- **Start conservative.** Bring a new layer onto a real robot with a low
  `amplitude`/`severity` first (a "nudge," not a hard wall) and increase
  it once you've watched it behave correctly around a person or obstacle a
  few times at low speed, rather than trusting simulation tuning directly
  at full speed on hardware.
