"""Single source of truth for AHA task categories.

Tasks fall into three buckets:
  - "training"  : the ~78 RLBench tasks used to generate the main AHA dataset
                  (rlbench-failgen/examples/ex_custom_data_generator.sh)
  - "held-out"  : the 10 evaluation tasks the model never saw in training
                  (rlbench-failgen/examples/ex_data_generator_eval.sh)
  - "extra"     : any other task that has a failgen config but is in neither set

Import `category_label(task_name)` to get a ready-to-print, colored label, or
`get_task_category(task_name)` for the raw category string.
"""

from aha_publish import paths

# Tasks used to generate the main AHA training dataset.
TRAINING_TASKS = {
    "basketball_in_hoop", "beat_the_buzz", "change_channel", "change_clock",
    "close_box", "close_door", "close_drawer", "close_fridge", "close_grill",
    "close_jar", "close_laptop_lid", "close_microwave", "get_ice_from_fridge",
    "hang_frame_on_hanger", "hit_ball_with_queue", "insert_onto_square_peg",
    "insert_usb_in_computer", "lamp_off", "move_hanger",
    "open_door", "open_drawer", "open_jar", "open_microwave",
    "open_window", "open_wine_bottle", "phone_on_base",
    "pick_and_lift", "place_shape_in_shape_sorter",
    "play_jenga", "plug_charger_in_power_supply", "pour_from_cup_to_cup",
    "press_switch", "push_buttons", "push_button", "put_bottle_in_fridge",
    "put_groceries_in_cupboard", "put_item_in_drawer", "put_knife_in_knife_block",
    "put_knife_on_chopping_board", "put_money_in_safe", "put_toilet_roll_on_stand",
    "reach_and_drag", "remove_cups", "scoop_with_spatula", "screw_nail",
    "setup_checkers", "setup_chess", "slide_block_to_target", "solve_puzzle",
    "stack_blocks", "stack_cups", "stack_wine", "straighten_rope",
    "sweep_to_dustpan", "take_frame_off_hanger",
    "take_item_out_of_drawer", "take_lid_off_saucepan", "take_money_out_safe",
    "take_off_weighing_scales", "take_plate_off_colored_dish_rack",
    "take_shoes_out_of_box", "take_toilet_roll_off_stand", "take_tray_out_of_oven",
    "put_shoes_in_box",
    "take_umbrella_out_of_umbrella_stand", "take_usb_out_of_computer",
    "toilet_seat_down", "toilet_seat_up", "turn_oven_on", "turn_tap", "tv_on",
    "unplug_charger", "water_plants", "wipe_desk",
}

# Held-out evaluation tasks (never seen during training).
HELD_OUT_TASKS = {
    "place_hanger_on_rack", "place_cups", "light_bulb_out",
    "light_bulb_in", "lamp_on", "hockey", "open_oven", "meat_on_grill",
    "pick_up_cup",
}

# ANSI colors for terminal output.
_COLORS = {
    "training": "\033[92m",   # green
    "held-out": "\033[94m",   # blue
    "extra": "\033[93m",      # yellow
}
_RESET = "\033[0m"


# Tasks where the robot sequentially interacts with exactly 2 distinct objects
# (e.g. first opens a container/door, then picks up and places a different object).
# The BT maker injects extra per-waypoint object-awareness guidance for these.
TWO_OBJECT_INTERACTION_TASKS = {
    "put_bottle_in_fridge",
    "put_item_in_drawer",
    "put_tray_in_oven",
    "slide_cabinet_open_and_place_cups",
    "stack_cups",
    "take_item_out_of_drawer",
    "take_tray_out_of_oven",
    "tv_on",
}

# Tasks where the robot sequentially interacts with 3 distinct objects
# (e.g. open a container, pick an object, place it inside — three separate targets).
THREE_OBJECT_INTERACTION_TASKS = {
    "put_shoes_in_box",
    "take_shoes_out_of_box",
}

# Tasks consisting of 5 independent single-object interactions chained in sequence
# (pick and place 5 separate items one after another).
# Each sub-task follows standard single-interaction rules; the gripper is open between sub-tasks.
# Currently empty: set_the_table, the only member, was removed from the benchmark.
FIVE_INTERACTION_TASKS = set()


def is_two_object_interaction(task_name):
    """Return True when the task involves sequential interaction with 2 distinct objects."""
    return task_name in TWO_OBJECT_INTERACTION_TASKS


def is_three_object_interaction(task_name):
    """Return True when the task involves sequential interaction with 3 distinct objects."""
    return task_name in THREE_OBJECT_INTERACTION_TASKS


def is_five_interaction(task_name):
    """Return True when the task consists of 5 independent sequential single-object interactions."""
    return task_name in FIVE_INTERACTION_TASKS


def get_task_category(task_name):
    """Return 'training', 'held-out', or 'extra' for the given task name."""
    if task_name in TRAINING_TASKS:
        return "training"
    if task_name in HELD_OUT_TASKS:
        return "held-out"
    return "extra"


def category_label(task_name, color=True):
    """Return a printable label like '[TRAINING]' for the task's category."""
    category = get_task_category(task_name)
    text = "[%s]" % category.upper()
    if color:
        return "%s%s%s" % (_COLORS.get(category, ""), text, _RESET)
    return text


def print_task_category(task_name, color=True):
    """Print the task name with its AHA category label."""
    print("AHA category: %s %s" % (task_name, category_label(task_name, color)))
