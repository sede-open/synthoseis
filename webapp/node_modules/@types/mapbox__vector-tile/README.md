# Installation
> `npm install --save @types/mapbox__vector-tile`

# Summary
This package contains type definitions for @mapbox/vector-tile (https://github.com/mapbox/vector-tile-js).

# Details
Files were exported from https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/mapbox__vector-tile.
## [index.d.ts](https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/mapbox__vector-tile/index.d.ts)
````ts
import Pbf = require("pbf");
import { Feature } from "geojson";
import Point = require("@mapbox/point-geometry");

export class VectorTile {
    constructor(pbf: Pbf);
    layers: { [_: string]: VectorTileLayer };
}

export class VectorTileFeature {
    static types: ["Unknown", "Point", "LineString", "Polygon"];
    extent: number;
    type: 0 | 1 | 2 | 3;
    id: number;
    properties: { [_: string]: string | number | boolean };
    loadGeometry(): Point[][];
    toGeoJSON(x: number, y: number, z: number): Feature;
    bbox?(): [number, number, number, number];
}

export class VectorTileLayer {
    constructor(pbf: Pbf);
    version?: number;
    name: string;
    extent: number;
    length: number;
    feature(featureIndex: number): VectorTileFeature;
}

````

### Additional Details
 * Last updated: Tue, 07 Nov 2023 09:09:39 GMT
 * Dependencies: [@types/geojson](https://npmjs.com/package/@types/geojson), [@types/mapbox__point-geometry](https://npmjs.com/package/@types/mapbox__point-geometry), [@types/pbf](https://npmjs.com/package/@types/pbf)

# Credits
These definitions were written by [Mathieu Maes](https://github.com/webberig), and [Harel Mazor](https://github.com/HarelM).
