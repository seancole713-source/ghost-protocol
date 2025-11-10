_ACTIVE = False


async def is_active():
    return _ACTIVE


async def set_active(v: bool):
    global _ACTIVE
    _ACTIVE = bool(v)
    return _ACTIVE
