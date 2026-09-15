"""Failure options available in an AHA task YAML."""

def get_task_waypoints(config):
    return list(config.get('data', {}).get('waypoints', []))

def get_available_failures(config):
    failures = [f for f in config.get('failures', []) if f.get('type') != 'dummy']
    has_freezing = any(f.get('type') == 'freezing' for f in failures)
    if not has_freezing:
        failures.append({
            'type': 'freezing',
            'name': 'freezing_random_midway',
            'enabled': False,
            'waypoints': get_task_waypoints(config),
            'freeze_after_range': [3, 10],
            'freeze_duration_seconds': 2.0,
        })
    return failures
