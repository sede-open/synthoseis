import { Array, Group, Location } from "./hierarchy.js";
import { jsonEncodeObject } from "./util.js";
export async function create(location, options = {}) {
    let loc = "store" in location ? location : new Location(location);
    if ("shape" in options) {
        let arr = await createArray(loc, options);
        return arr;
    }
    return createGroup(loc, options);
}
async function createGroup(location, options = {}) {
    let metadata = {
        zarr_format: 3,
        node_type: "group",
        attributes: options.attributes ?? {},
    };
    await location.store.set(location.resolve("zarr.json").path, jsonEncodeObject(metadata));
    return new Group(location.store, location.path, metadata);
}
async function createArray(location, options) {
    let metadata = {
        zarr_format: 3,
        node_type: "array",
        shape: options.shape,
        data_type: options.dtype,
        chunk_grid: {
            name: "regular",
            configuration: {
                chunk_shape: options.chunkShape,
            },
        },
        chunk_key_encoding: {
            name: "default",
            configuration: {
                separator: options.chunkSeparator ?? "/",
            },
        },
        codecs: options.codecs ?? [],
        fill_value: options.fillValue ?? null,
        dimension_names: options.dimensionNames,
        attributes: options.attributes ?? {},
    };
    await location.store.set(location.resolve("zarr.json").path, jsonEncodeObject(metadata));
    return new Array(location.store, location.path, metadata);
}
//# sourceMappingURL=create.js.map