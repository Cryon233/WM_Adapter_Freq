import math
import os
import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache

import numpy as np
from robosuite.utils.mjcf_utils import find_elements, string_to_array

import robocasa
from robocasa.models.objects.kitch_min_obj import OBJ_CATEGORIES, OBJ_GROUPS

BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")
_REQUIRED_GEOMETRY_SITES = (
    "bottom_site",
    "top_site",
    "horizontal_radius_site",
)
_MAX_OBJECT_SAMPLE_ATTEMPTS = 4096
_MJCF_GEOMETRY_FAILURES = {}


def _load_mjcf_geometry(
    mjcf_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    normalized_path = os.path.realpath(os.path.abspath(mjcf_path))
    return _load_mjcf_geometry_cached(normalized_path)


@lru_cache(maxsize=None)
def _load_mjcf_geometry_cached(
    mjcf_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        tree = ET.parse(mjcf_path)
    except (ET.ParseError, OSError) as exc:
        _MJCF_GEOMETRY_FAILURES[mjcf_path] = "parse_error"
        warnings.warn(
            f"Skipping incompatible RoboCasa object {mjcf_path}: "
            f"failed to parse MJCF ({exc})",
            RuntimeWarning,
        )
        return None

    root = tree.getroot()
    site_elements = {
        name: find_elements(
            root=root,
            tags="site",
            attribs={"name": name},
        )
        for name in _REQUIRED_GEOMETRY_SITES
    }
    missing_sites = [
        name
        for name, element in site_elements.items()
        if element is None or element.get("pos") is None
    ]
    if missing_sites:
        _MJCF_GEOMETRY_FAILURES[mjcf_path] = "missing_geometry"
        warnings.warn(
            f"Skipping incompatible RoboCasa object {mjcf_path}: "
            f"missing required MJCF sites {missing_sites}",
            RuntimeWarning,
        )
        return None

    try:
        geometry = (
            string_to_array(site_elements["bottom_site"].get("pos")),
            string_to_array(site_elements["top_site"].get("pos")),
            string_to_array(site_elements["horizontal_radius_site"].get("pos")),
        )
    except (TypeError, ValueError) as exc:
        _MJCF_GEOMETRY_FAILURES[mjcf_path] = "missing_geometry"
        warnings.warn(
            f"Skipping incompatible RoboCasa object {mjcf_path}: "
            f"invalid required MJCF site positions ({exc})",
            RuntimeWarning,
        )
        return None

    invalid_sites = [
        name
        for name, position in zip(_REQUIRED_GEOMETRY_SITES, geometry)
        if position.size < 3 or not np.isfinite(position).all()
    ]
    if invalid_sites:
        _MJCF_GEOMETRY_FAILURES[mjcf_path] = "missing_geometry"
        warnings.warn(
            f"Skipping incompatible RoboCasa object {mjcf_path}: "
            f"invalid required MJCF site positions {invalid_sites}",
            RuntimeWarning,
        )
        return None
    return geometry


def _mjcf_geometry_failure(mjcf_path: str) -> str | None:
    normalized_path = os.path.realpath(os.path.abspath(mjcf_path))
    return _MJCF_GEOMETRY_FAILURES.get(normalized_path)


class ObjCat:
    """
    Class that encapsulates data for an object category.

    Args:
        name (str): name of the object category

        types (tuple) or (str): type(s)/categories the object belongs to. Examples include meat, sweets, fruit, etc.

        model_folders (list): list of folders containing the MJCF models for the object category

        exclude (list): list of model names to exclude

        graspable (bool): whether the object is graspable

        washable (bool): whether the object is washable

        microwavable (bool): whether the object is microwavable

        cookable (bool): whether the object is cookable

        freezable (bool): whether the object is freezable

        scale (float): scale of the object meshes/geoms

        solimp (tuple): solimp values for the object meshes/geoms

        solref (tuple): solref values for the object meshes/geoms

        density (float): density of the object meshes/geoms

        friction (tuple): friction values for the object meshes/geoms

        priority: priority of the object

        aigen_cat (bool): True if the object is an AI-generated object otherwise its an objaverse object
    """

    def __init__(
        self,
        name,
        types,
        model_folders=None,
        exclude=None,
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 2),
        density=50,
        friction=(2.95, 3.3, 1.1),
        priority=None,
        aigen_cat=True, #BUG CAUSE: set to False, to be able to create an env, when not having downloaded aigen assets
    ):
        self.name = name
        if not isinstance(types, tuple):
            types = (types,)
        self.types = types

        self.aigen_cat = aigen_cat

        self.graspable = graspable
        self.washable = washable
        self.microwavable = microwavable
        self.cookable = cookable
        self.freezable = freezable

        self.scale = scale
        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.priority = priority
        self.exclude = exclude or []

        if model_folders is None:
            subf = "aigen_objs" if self.aigen_cat else "objaverse"
            model_folders = ["{}/{}".format(subf, name)]
        cat_mjcf_paths = []
        for folder in model_folders:
            cat_path = os.path.join(BASE_ASSET_ZOO_PATH, folder)
            for root, _, files in os.walk(cat_path):
                if "model.xml" in files:
                    model_name = os.path.basename(root)
                    if model_name in self.exclude:
                        continue
                    cat_mjcf_paths.append(os.path.join(root, "model.xml"))
        self.mjcf_paths = sorted(cat_mjcf_paths)
        # if cat_mjcf_paths == []: breakpoint()

    def get_mjcf_kwargs(self):
        """
        returns relevant data to apply to the MJCF model for the object category
        """
        return deepcopy(
            dict(
                scale=self.scale,
                solimp=self.solimp,
                solref=self.solref,
                density=self.density,
                friction=self.friction,
                priority=self.priority,
            )
        )


# update OBJ_CATEGORIES with ObjCat instances. Maps name to the different registries it can belong to
# and then maps the registry to the ObjCat instance
for (name, kwargs) in OBJ_CATEGORIES.items():

    # get the properties that are common to both registries
    common_properties = deepcopy(kwargs)
    for k in common_properties.keys():
        assert k in [
            "graspable",
            "washable",
            "microwavable",
            "cookable",
            "freezable",
            "types",
            "aigen",
            "objaverse",
        ]
    objaverse_kwargs = common_properties.pop("objaverse", None)
    aigen_kwargs = common_properties.pop("aigen", None)
    assert "scale" not in kwargs
    OBJ_CATEGORIES[name] = {}

    # create instances
    # breakpoint()
    if objaverse_kwargs is not None:
        objaverse_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["objaverse"] = ObjCat(name=name, **objaverse_kwargs)
        large_obj = [
            "liquor",
            "jug",
            "bottled_water",
            "boxed_food",
            "cereal",
            "milk",
            "spray",
            "teapot",
            "jug",
            "coconut",
            "lemonade",
            "squash",
        ]
        if OBJ_CATEGORIES[name]["objaverse"].name in large_obj:
            OBJ_CATEGORIES[name]["objaverse"].scale *= 1.25
            OBJ_CATEGORIES[name]["objaverse"].density = 25
        else:
            OBJ_CATEGORIES[name]["objaverse"].scale *= 1.5
            OBJ_CATEGORIES[name]["objaverse"].density = 25

    if aigen_kwargs is not None:
        aigen_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["aigen"] = ObjCat(
            name=name, aigen_cat=True, **aigen_kwargs
        )


def sample_kitchen_object(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    freezable=None,
    rng=None,
    obj_registries=("objaverse",),
    split=None,
    max_size=(None, None, None),
    object_scale=None,
    sampled_objects=None,
):
    """
    Sample a kitchen object from the specified groups and within max_size bounds.

    Args:
        groups (list or str): groups to sample from or the exact xml path of the object to spawn

        exclude_groups (str or list): groups to exclude

        graspable (bool): whether the sampled object must be graspable

        washable (bool): whether the sampled object must be washable

        microwavable (bool): whether the sampled object must be microwavable

        cookable (bool): whether whether the sampled object must be cookable

        freezable (bool): whether whether the sampled object must be freezable

        rng (np.random.Generator): random number object

        obj_registries (tuple): registries to sample from

        split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                    (or the first half - whichever is larger), "B" specifies the  rest, and None specifies all.

        max_size (tuple): max size of the object. If the sampled object is not within bounds of max size, function will resample

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """
    if sampled_objects is None:
        sampled_objects = set()
    rejected_missing_geometry = 0
    rejected_size = 0
    rejected_parse_error = 0
    for attempt in range(1, _MAX_OBJECT_SAMPLE_ATTEMPTS + 1):
        mjcf_kwargs, info = sample_kitchen_object_helper(
            groups=groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=rng,
            obj_registries=obj_registries,
            split=split,
            object_scale=object_scale,
        )

        # check if object size is within bounds
        mjcf_path = info["mjcf_path"]
        if mjcf_path in sampled_objects:
            continue  # Skip this object and sample again

        # Add the object to the set of sampled objects
        sampled_objects.add(mjcf_path)

        geometry = _load_mjcf_geometry(str(mjcf_path))
        if geometry is None:
            failure = _mjcf_geometry_failure(str(mjcf_path))
            if failure == "parse_error":
                rejected_parse_error += 1
            else:
                rejected_missing_geometry += 1
            if isinstance(groups, str) and groups.endswith(".xml"):
                raise RuntimeError(
                    "Explicit RoboCasa object has incompatible MJCF geometry: "
                    f"{mjcf_path}"
                )
            continue
        bottom, top, horizontal_radius = geometry
        scale = mjcf_kwargs["scale"]
        obj_size = (
            np.array(
                [horizontal_radius[0] * 2, horizontal_radius[1] * 2, top[2] - bottom[2]]
            )
            * scale
        )
        if all(max_size[i] is None or obj_size[i] <= max_size[i] for i in range(3)):
            return mjcf_kwargs, info
        rejected_size += 1

    raise RuntimeError(
        "Unable to sample a size-compatible RoboCasa object after bounded attempts: "
        f"obj_groups={groups!r}, exclude_obj_groups={exclude_groups!r}, "
        f"registries={obj_registries!r}, split={split!r}, max_size={max_size!r}, "
        f"attempts={_MAX_OBJECT_SAMPLE_ATTEMPTS}, "
        f"rejected_missing_geometry={rejected_missing_geometry}, "
        f"rejected_size={rejected_size}, rejected_parse_error={rejected_parse_error}, "
        f"unique_sampled_paths={len(sampled_objects)}"
    )


def sample_kitchen_object_helper(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    freezable=None,
    rng=None,
    obj_registries=("objaverse",),
    split=None,
    object_scale=None,
):
    """
    Helper function to sample a kitchen object.

    Args:
        groups (list or str): groups to sample from or the exact xml path of the object to spawn

        exclude_groups (str or list): groups to exclude

        graspable (bool): whether the sampled object must be graspable

        washable (bool): whether the sampled object must be washable

        microwavable (bool): whether the sampled object must be microwavable

        cookable (bool): whether whether the sampled object must be cookable

        freezable (bool): whether whether the sampled object must be freezable

        rng (np.random.Generator): random number object

        obj_registries (tuple): registries to sample from

        split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                    (or the first half - whichever is larger), "B" specifies the  rest, and None specifies all.

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """
    if rng is None:
        rng = np.random.default_rng()

    # option to spawn specific object instead of sampling from a group
    if isinstance(groups, str) and groups.endswith(".xml"):
        mjcf_path = groups
        if _load_mjcf_geometry(str(mjcf_path)) is None:
            raise RuntimeError(
                f"Explicit RoboCasa object has incompatible MJCF geometry: {mjcf_path}"
            )
        # reverse look up mjcf_path to category
        mjcf_kwargs = dict()
        cat = None
        obj_found = False
        for cand_cat in OBJ_CATEGORIES:
            for reg in obj_registries:
                if reg in OBJ_CATEGORIES[cand_cat]:
                    mjcf_kwargs = OBJ_CATEGORIES[cand_cat][reg].get_mjcf_kwargs()
                    cat = cand_cat
                    obj_found = True
                    break
            if obj_found:
                break
        if obj_found is False:
            raise ValueError
        mjcf_kwargs["mjcf_path"] = mjcf_path
    else:
        if not isinstance(groups, tuple) and not isinstance(groups, list):
            groups = [groups]

        if exclude_groups is None:
            exclude_groups = []
        if not isinstance(exclude_groups, tuple) and not isinstance(
            exclude_groups, list
        ):
            exclude_groups = [exclude_groups]

        invalid_categories = []
        for g in exclude_groups:
            for cat in OBJ_GROUPS[g]:
                invalid_categories.append(cat)

        valid_categories = []
        for g in groups:
            for cat in OBJ_GROUPS[g]:
                # don't repeat if already added
                if cat in valid_categories:
                    continue
                if cat in invalid_categories:
                    continue

                # don't include if category not represented in any registry
                cat_in_any_reg = np.any(
                    [reg in OBJ_CATEGORIES[cat] for reg in obj_registries]
                )
                if not cat_in_any_reg:
                    continue

                invalid = False
                for reg in obj_registries:
                    if reg not in OBJ_CATEGORIES[cat]:
                        continue
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if graspable is True and cat_meta.graspable is not True:
                        invalid = True
                    if washable is True and cat_meta.washable is not True:
                        invalid = True
                    if microwavable is True and cat_meta.microwavable is not True:
                        invalid = True
                    if cookable is True and cat_meta.cookable is not True:
                        invalid = True
                    if freezable is True and cat_meta.freezable is not True:
                        invalid = True

                if invalid:
                    continue

                valid_categories.append(cat)

        cat = rng.choice(valid_categories)
        choices = {reg: [] for reg in obj_registries}

        for reg in obj_registries:
            if reg not in OBJ_CATEGORIES[cat]:
                choices[reg] = []
                continue
            reg_choices = deepcopy(OBJ_CATEGORIES[cat][reg].mjcf_paths)

            # exclude out objects based on split
            if split is not None:
                split_th = max(len(choices) - 3, int(math.ceil(len(reg_choices) / 2)))
                if split == "A":
                    reg_choices = reg_choices[:split_th]
                elif split == "B":
                    reg_choices = reg_choices[split_th:]
                else:
                    raise ValueError
            choices[reg] = reg_choices

        chosen_reg = rng.choice(
            obj_registries,
            p=np.array([len(choices[reg]) for reg in obj_registries])
            / sum(len(choices[reg]) for reg in obj_registries),
        )

        mjcf_path = rng.choice(choices[chosen_reg])
        mjcf_kwargs = OBJ_CATEGORIES[cat][chosen_reg].get_mjcf_kwargs()
        mjcf_kwargs["mjcf_path"] = mjcf_path

    if object_scale is not None:
        mjcf_kwargs["scale"] *= object_scale

    groups_containing_sampled_obj = []
    for group, group_cats in OBJ_GROUPS.items():
        if cat in group_cats:
            groups_containing_sampled_obj.append(group)

    info = {
        "groups_containing_sampled_obj": groups_containing_sampled_obj,
        "groups": groups,
        "cat": cat,
        "split": split,
        "mjcf_path": mjcf_path,
    }

    return mjcf_kwargs, info
