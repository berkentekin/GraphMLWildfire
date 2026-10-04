BASE_FEATURE_NAMES = (
    "viirs_m11",
    "viirs_i2",
    "viirs_i1",
    "ndvi",
    "evi2",
    "total_precipitation",
    "wind_speed",
    "wind_direction",
    "minimum_temperature",
    "maximum_temperature",
    "energy_release_component",
    "specific_humidity",
    "slope",
    "aspect",
    "elevation",
    "pdsi",
    "land_cover",
    "forecast_total_precipitation",
    "forecast_wind_speed",
    "forecast_wind_direction",
    "forecast_temperature",
    "forecast_specific_humidity",
    "active_fire",
)

ANGLE_INDICES = (7, 13, 19)
LAND_COVER_INDEX = 16
ACTIVE_FIRE_INDEX = 22
LAND_COVER_CLASSES = 17

PHYSICS_NAMES = (
    "wind_speed",
    "wind_direction",
    "forecast_wind_speed",
    "forecast_wind_direction",
    "elevation",
    "slope",
    "aspect",
)


def processed_feature_names() -> tuple[str, ...]:
    names = []
    for index, name in enumerate(BASE_FEATURE_NAMES):
        if index in ANGLE_INDICES:
            names.extend((f"{name}_sin", f"{name}_cos"))
        elif index == LAND_COVER_INDEX:
            names.extend(f"land_cover_{class_id}" for class_id in range(1, 18))
            names.append("land_cover_unknown")
        elif index != ACTIVE_FIRE_INDEX:
            names.append(name)
    return tuple(names)


PROCESSED_FEATURE_NAMES = processed_feature_names()
CONTINUOUS_FEATURE_INDICES = tuple(
    index
    for index, name in enumerate(PROCESSED_FEATURE_NAMES)
    if not name.endswith(("_sin", "_cos")) and not name.startswith("land_cover_")
)
