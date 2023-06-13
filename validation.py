import utils

def part_name_control(part_name, landmark_path):
    """
    checks part_name. if it is all, returns True. Otherwise it is False
    """
    landmarks = utils.get_yaml(landmark_path)

    if part_name == 'all':
        return True

    elif part_name not in landmarks:
        raise ValueError("part_name parameter can be one of them face, lips, eye, eyebrow")
    
    return False