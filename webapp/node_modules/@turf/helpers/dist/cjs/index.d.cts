import { Feature, Point, Position, LineString, MultiLineString, Polygon, MultiPolygon, FeatureCollection, Geometry, GeometryCollection, GeometryObject, GeoJsonProperties, BBox, MultiPoint } from 'geojson';

/**
 * Id
 *
 * https://tools.ietf.org/html/rfc7946#section-3.2
 * If a Feature has a commonly used identifier, that identifier SHOULD be included as a member of
 * the Feature object with the name "id", and the value of this member is either a JSON string or number.
 *
 * Should be contributed to @types/geojson
 */
type Id = string | number;

/**
 * @module helpers
 */
type Coord = Feature<Point> | Point | Position;
/**
 * Linear measurement units.
 *
 * ⚠️ Warning. Be aware of the implications of using radian or degree units to
 * measure distance. The distance represented by a degree of longitude *varies*
 * depending on latitude.
 *
 * See https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616
 * for an illustration of this behaviour.
 *
 * @typedef
 */
type Units = "meters" | "metres" | "millimeters" | "millimetres" | "centimeters" | "centimetres" | "kilometers" | "kilometres" | "miles" | "nauticalmiles" | "inches" | "yards" | "feet" | "radians" | "degrees";
/**
 * Area measurement units.
 *
 * @typedef
 */
type AreaUnits = Exclude<Units, "radians" | "degrees"> | "acres" | "hectares";
/**
 * Grid types.
 *
 * @typedef
 */
type Grid = "point" | "square" | "hex" | "triangle";
/**
 * Shorthand corner identifiers.
 *
 * @typedef
 */
type Corners = "sw" | "se" | "nw" | "ne" | "center" | "centroid";
/**
 * Geometries made up of lines i.e. lines and polygons.
 *
 * @typedef
 */
type Lines = LineString | MultiLineString | Polygon | MultiPolygon;
/**
 * Convenience type for all possible GeoJSON.
 *
 * @typedef
 */
type AllGeoJSON = Feature | FeatureCollection | Geometry | GeometryCollection;
/**
 * The Earth radius in meters. Used by Turf modules that model the Earth as a sphere. The {@link https://en.wikipedia.org/wiki/Earth_radius#Arithmetic_mean_radius mean radius} was selected because it is {@link https://rosettacode.org/wiki/Haversine_formula#:~:text=This%20value%20is%20recommended recommended } by the Haversine formula (used by turf/distance) to reduce error.
 *
 * @constant
 */
declare const earthRadius = 6371008.8;
/**
 * Unit of measurement factors based on earthRadius.
 *
 * Keys are the name of the unit, values are the number of that unit in a single radian
 *
 * @constant
 */
declare const factors: Record<Units, number>;
/**

 * Area of measurement factors based on 1 square meter.
 *
 * @constant
 */
declare const areaFactors: Record<AreaUnits, number>;
/**
 * Wraps a GeoJSON {@link Geometry} in a GeoJSON {@link Feature}.
 *
 * @function
 * @param {GeometryObject} geometry input geometry
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<GeometryObject, GeoJsonProperties>} a GeoJSON Feature
 * @example
 * var geometry = {
 *   "type": "Point",
 *   "coordinates": [110, 50]
 * };
 *
 * var feature = turf.feature(geometry);
 *
 * //=feature
 */
declare function feature<G extends GeometryObject = Geometry, P extends GeoJsonProperties = GeoJsonProperties>(geom: G | null, properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<G, P>;
/**
 * Creates a GeoJSON {@link Geometry} from a Geometry string type & coordinates.
 * For GeometryCollection type use `helpers.geometryCollection`
 *
 * @function
 * @param {("Point" | "LineString" | "Polygon" | "MultiPoint" | "MultiLineString" | "MultiPolygon")} type Geometry Type
 * @param {Array<any>} coordinates Coordinates
 * @param {Object} [options={}] Optional Parameters
 * @returns {Geometry} a GeoJSON Geometry
 * @example
 * var type = "Point";
 * var coordinates = [110, 50];
 * var geometry = turf.geometry(type, coordinates);
 * // => geometry
 */
declare function geometry<T extends "Point" | "LineString" | "Polygon" | "MultiPoint" | "MultiLineString" | "MultiPolygon">(type: T, coordinates: any[], _options?: Record<string, never>): Extract<Geometry, {
    type: T;
}>;
/**
 * Creates a {@link Point} {@link Feature} from a Position.
 *
 * @function
 * @param {Position} coordinates longitude, latitude position (each in decimal degrees)
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<Point, GeoJsonProperties>} a Point feature
 * @example
 * var point = turf.point([-75.343, 39.984]);
 *
 * //=point
 */
declare function point<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position, properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<Point, P>;
/**
 * Creates a {@link Point} {@link FeatureCollection} from an Array of Point coordinates.
 *
 * @function
 * @param {Position[]} coordinates an array of Points
 * @param {GeoJsonProperties} [properties={}] Translate these properties to each Feature
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north]
 * associated with the FeatureCollection
 * @param {Id} [options.id] Identifier associated with the FeatureCollection
 * @returns {FeatureCollection<Point>} Point Feature
 * @example
 * var points = turf.points([
 *   [-75, 39],
 *   [-80, 45],
 *   [-78, 50]
 * ]);
 *
 * //=points
 */
declare function points<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): FeatureCollection<Point, P>;
/**
 * Creates a {@link Polygon} {@link Feature} from an Array of LinearRings.
 *
 * @function
 * @param {Position[][]} coordinates an array of LinearRings
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<Polygon, GeoJsonProperties>} Polygon Feature
 * @example
 * var polygon = turf.polygon([[[-5, 52], [-4, 56], [-2, 51], [-7, 54], [-5, 52]]], { name: 'poly1' });
 *
 * //=polygon
 */
declare function polygon<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[][], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<Polygon, P>;
/**
 * Creates a {@link Polygon} {@link FeatureCollection} from an Array of Polygon coordinates.
 *
 * @function
 * @param {Position[][][]} coordinates an array of Polygon coordinates
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the FeatureCollection
 * @returns {FeatureCollection<Polygon, GeoJsonProperties>} Polygon FeatureCollection
 * @example
 * var polygons = turf.polygons([
 *   [[[-5, 52], [-4, 56], [-2, 51], [-7, 54], [-5, 52]]],
 *   [[[-15, 42], [-14, 46], [-12, 41], [-17, 44], [-15, 42]]],
 * ]);
 *
 * //=polygons
 */
declare function polygons<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[][][], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): FeatureCollection<Polygon, P>;
/**
 * Creates a {@link LineString} {@link Feature} from an Array of Positions.
 *
 * @function
 * @param {Position[]} coordinates an array of Positions
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<LineString, GeoJsonProperties>} LineString Feature
 * @example
 * var linestring1 = turf.lineString([[-24, 63], [-23, 60], [-25, 65], [-20, 69]], {name: 'line 1'});
 * var linestring2 = turf.lineString([[-14, 43], [-13, 40], [-15, 45], [-10, 49]], {name: 'line 2'});
 *
 * //=linestring1
 * //=linestring2
 */
declare function lineString<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<LineString, P>;
/**
 * Creates a {@link LineString} {@link FeatureCollection} from an Array of LineString coordinates.
 *
 * @function
 * @param {Position[][]} coordinates an array of LinearRings
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north]
 * associated with the FeatureCollection
 * @param {Id} [options.id] Identifier associated with the FeatureCollection
 * @returns {FeatureCollection<LineString, GeoJsonProperties>} LineString FeatureCollection
 * @example
 * var linestrings = turf.lineStrings([
 *   [[-24, 63], [-23, 60], [-25, 65], [-20, 69]],
 *   [[-14, 43], [-13, 40], [-15, 45], [-10, 49]]
 * ]);
 *
 * //=linestrings
 */
declare function lineStrings<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[][], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): FeatureCollection<LineString, P>;
/**
 * Takes one or more {@link Feature|Features} and creates a {@link FeatureCollection}.
 *
 * @function
 * @param {Array<Feature<GeometryObject, GeoJsonProperties>>} features input features
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {FeatureCollection<GeometryObject, GeoJsonProperties>} FeatureCollection of Features
 * @example
 * var locationA = turf.point([-75.343, 39.984], {name: 'Location A'});
 * var locationB = turf.point([-75.833, 39.284], {name: 'Location B'});
 * var locationC = turf.point([-75.534, 39.123], {name: 'Location C'});
 *
 * var collection = turf.featureCollection([
 *   locationA,
 *   locationB,
 *   locationC
 * ]);
 *
 * //=collection
 */
declare function featureCollection<G extends GeometryObject = Geometry, P extends GeoJsonProperties = GeoJsonProperties>(features: Array<Feature<G, P>>, options?: {
    bbox?: BBox;
    id?: Id;
}): FeatureCollection<G, P>;
/**
 * Creates a {@link Feature}<{@link MultiLineString}> based on a
 * coordinate array. Properties can be added optionally.
 *
 * @function
 * @param {Position[][]} coordinates an array of LineStrings
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<MultiLineString, GeoJsonProperties>} a MultiLineString feature
 * @throws {Error} if no coordinates are passed
 * @example
 * var multiLine = turf.multiLineString([[[0,0],[10,10]]]);
 *
 * //=multiLine
 */
declare function multiLineString<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[][], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<MultiLineString, P>;
/**
 * Creates a {@link Feature}<{@link MultiPoint}> based on a
 * coordinate array. Properties can be added optionally.
 *
 * @function
 * @param {Position[]} coordinates an array of Positions
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<MultiPoint, GeoJsonProperties>} a MultiPoint feature
 * @throws {Error} if no coordinates are passed
 * @example
 * var multiPt = turf.multiPoint([[0,0],[10,10]]);
 *
 * //=multiPt
 */
declare function multiPoint<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<MultiPoint, P>;
/**
 * Creates a {@link Feature}<{@link MultiPolygon}> based on a
 * coordinate array. Properties can be added optionally.
 *
 * @function
 * @param {Position[][][]} coordinates an array of Polygons
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<MultiPolygon, GeoJsonProperties>} a multipolygon feature
 * @throws {Error} if no coordinates are passed
 * @example
 * var multiPoly = turf.multiPolygon([[[[0,0],[0,10],[10,10],[10,0],[0,0]]]]);
 *
 * //=multiPoly
 *
 */
declare function multiPolygon<P extends GeoJsonProperties = GeoJsonProperties>(coordinates: Position[][][], properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<MultiPolygon, P>;
/**
 * Creates a Feature<GeometryCollection> based on a
 * coordinate array. Properties can be added optionally.
 *
 * @function
 * @param {Array<Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon>} geometries an array of GeoJSON Geometries
 * @param {GeoJsonProperties} [properties={}] an Object of key-value pairs to add as properties
 * @param {Object} [options={}] Optional Parameters
 * @param {BBox} [options.bbox] Bounding Box Array [west, south, east, north] associated with the Feature
 * @param {Id} [options.id] Identifier associated with the Feature
 * @returns {Feature<GeometryCollection, GeoJsonProperties>} a GeoJSON GeometryCollection Feature
 * @example
 * var pt = turf.geometry("Point", [100, 0]);
 * var line = turf.geometry("LineString", [[101, 0], [102, 1]]);
 * var collection = turf.geometryCollection([pt, line]);
 *
 * // => collection
 */
declare function geometryCollection<G extends Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon, P extends GeoJsonProperties = GeoJsonProperties>(geometries: Array<G>, properties?: P, options?: {
    bbox?: BBox;
    id?: Id;
}): Feature<GeometryCollection<G>, P>;
/**
 * Round number to precision
 *
 * @function
 * @param {number} num Number
 * @param {number} [precision=0] Precision
 * @returns {number} rounded number
 * @example
 * turf.round(120.4321)
 * //=120
 *
 * turf.round(120.4321, 2)
 * //=120.43
 */
declare function round(num: number, precision?: number): number;
/**
 * Convert a distance measurement (assuming a spherical Earth) from radians to a more friendly unit.
 * Valid units: miles, nauticalmiles, inches, yards, meters, metres, kilometers, centimeters, feet
 *
 * @function
 * @param {number} radians in radians across the sphere
 * @param {Units} [units="kilometers"] can be degrees, radians, miles, inches, yards, metres,
 * meters, kilometres, kilometers.
 * @returns {number} distance
 */
declare function radiansToLength(radians: number, units?: Units): number;
/**
 * Convert a distance measurement (assuming a spherical Earth) from a real-world unit into radians
 * Valid units: miles, nauticalmiles, inches, yards, meters, metres, kilometers, centimeters, feet
 *
 * @function
 * @param {number} distance in real units
 * @param {Units} [units="kilometers"] can be degrees, radians, miles, inches, yards, metres,
 * meters, kilometres, kilometers.
 * @returns {number} radians
 */
declare function lengthToRadians(distance: number, units?: Units): number;
/**
 * Convert a distance measurement (assuming a spherical Earth) from a real-world unit into degrees
 * Valid units: miles, nauticalmiles, inches, yards, meters, metres, centimeters, kilometres, feet
 *
 * @function
 * @param {number} distance in real units
 * @param {Units} [units="kilometers"] can be degrees, radians, miles, inches, yards, metres,
 * meters, kilometres, kilometers.
 * @returns {number} degrees
 */
declare function lengthToDegrees(distance: number, units?: Units): number;
/**
 * Converts any bearing angle from the north line direction (positive clockwise)
 * and returns an angle between 0-360 degrees (positive clockwise), 0 being the north line
 *
 * @function
 * @param {number} bearing angle, between -180 and +180 degrees
 * @returns {number} angle between 0 and 360 degrees
 */
declare function bearingToAzimuth(bearing: number): number;
/**
 * Converts any azimuth angle from the north line direction (positive clockwise)
 * and returns an angle between -180 and +180 degrees (positive clockwise), 0 being the north line
 *
 * @function
 * @param {number} angle between 0 and 360 degrees
 * @returns {number} bearing between -180 and +180 degrees
 */
declare function azimuthToBearing(angle: number): number;
/**
 * Converts an angle in radians to degrees
 *
 * @function
 * @param {number} radians angle in radians
 * @returns {number} degrees between 0 and 360 degrees
 */
declare function radiansToDegrees(radians: number): number;
/**
 * Converts an angle in degrees to radians
 *
 * @function
 * @param {number} degrees angle between 0 and 360 degrees
 * @returns {number} angle in radians
 */
declare function degreesToRadians(degrees: number): number;
/**
 * Converts a length from one unit to another.
 *
 * @function
 * @param {number} length Length to be converted
 * @param {Units} [originalUnit="kilometers"] Input length unit
 * @param {Units} [finalUnit="kilometers"] Returned length unit
 * @returns {number} The converted length
 */
declare function convertLength(length: number, originalUnit?: Units, finalUnit?: Units): number;
/**
 * Converts an area from one unit to another.
 *
 * @function
 * @param {number} area Area to be converted
 * @param {AreaUnits} [originalUnit="meters"] Input area unit
 * @param {AreaUnits} [finalUnit="kilometers"] Returned area unit
 * @returns {number} The converted length
 */
declare function convertArea(area: number, originalUnit?: AreaUnits, finalUnit?: AreaUnits): number;
/**
 * isNumber
 *
 * @function
 * @param {any} num Number to validate
 * @returns {boolean} true/false
 * @example
 * turf.isNumber(123)
 * //=true
 * turf.isNumber('foo')
 * //=false
 */
declare function isNumber(num: any): boolean;
/**
 * isObject
 *
 * @function
 * @param {any} input variable to validate
 * @returns {boolean} true/false, including false for Arrays and Functions
 * @example
 * turf.isObject({elevation: 10})
 * //=true
 * turf.isObject('foo')
 * //=false
 */
declare function isObject(input: any): boolean;
/**
 * Validate BBox
 *
 * @private
 * @param {any} bbox BBox to validate
 * @returns {void}
 * @throws {Error} if BBox is not valid
 * @example
 * validateBBox([-180, -40, 110, 50])
 * //=OK
 * validateBBox([-180, -40])
 * //=Error
 * validateBBox('Foo')
 * //=Error
 * validateBBox(5)
 * //=Error
 * validateBBox(null)
 * //=Error
 * validateBBox(undefined)
 * //=Error
 */
declare function validateBBox(bbox: any): void;
/**
 * Validate Id
 *
 * @private
 * @param {any} id Id to validate
 * @returns {void}
 * @throws {Error} if Id is not valid
 * @example
 * validateId([-180, -40, 110, 50])
 * //=Error
 * validateId([-180, -40])
 * //=Error
 * validateId('Foo')
 * //=OK
 * validateId(5)
 * //=OK
 * validateId(null)
 * //=Error
 * validateId(undefined)
 * //=Error
 */
declare function validateId(id: any): void;

export { type AllGeoJSON, type AreaUnits, type Coord, type Corners, type Grid, type Id, type Lines, type Units, areaFactors, azimuthToBearing, bearingToAzimuth, convertArea, convertLength, degreesToRadians, earthRadius, factors, feature, featureCollection, geometry, geometryCollection, isNumber, isObject, lengthToDegrees, lengthToRadians, lineString, lineStrings, multiLineString, multiPoint, multiPolygon, point, points, polygon, polygons, radiansToDegrees, radiansToLength, round, validateBBox, validateId };
