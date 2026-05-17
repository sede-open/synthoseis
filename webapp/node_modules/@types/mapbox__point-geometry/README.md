# Installation
> `npm install --save @types/mapbox__point-geometry`

# Summary
This package contains type definitions for @mapbox/point-geometry (https://github.com/mapbox/point-geometry).

# Details
Files were exported from https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/mapbox__point-geometry.
## [index.d.ts](https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/mapbox__point-geometry/index.d.ts)
````ts
declare class Point {
    x: number;
    y: number;
    constructor(x: number, y: number);
    clone(): Point;
    add(p: Point): Point;
    _add(p: Point): Point;
    sub(p: Point): Point;
    _sub(p: Point): Point;
    multiByPoint(p: Point): Point;
    divByPoint(p: Point): Point;
    mult(k: number): Point;
    _mult(k: number): Point;
    div(k: number): Point;
    _div(k: number): Point;
    rotate(k: number): Point;
    _rotate(k: number): Point;
    rotateAround(k: number, p: Point): Point;
    _rotateAround(k: number, p: Point): Point;
    matMult(m: number[]): Point;
    _matMult(m: number[]): Point;
    unit(): Point;
    _unit(): Point;
    perp(): Point;
    _perp(): Point;
    round(): Point;
    _round(): Point;
    mag(): number;
    equals(other: Point): boolean;
    dist(p: Point): number;
    distSqr(p: Point): number;
    angle(): number;
    angleTo(b: Point): number;
    angleWith(b: Point): number;
    angleWithSep(x: number, y: number): number;

    static convert<T extends unknown>(a: T): T extends number[] ? Point : T extends Point ? Point : T;
}

export = Point;

````

### Additional Details
 * Last updated: Tue, 07 Nov 2023 09:09:39 GMT
 * Dependencies: none

# Credits
These definitions were written by [Mathieu Maes](https://github.com/webberig), and [Harel Mazor](https://github.com/HarelM).
