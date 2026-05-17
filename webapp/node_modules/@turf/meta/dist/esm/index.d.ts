import { GeoJsonProperties, Feature, FeatureCollection, GeometryCollection, Geometry, GeometryObject, BBox, LineString, MultiLineString, Polygon, MultiPolygon, Point } from 'geojson';
import { AllGeoJSON, Id, Lines } from '@turf/helpers';

/**
 * Callback for coordEach
 *
 * @callback coordEachCallback
 * @param {number[]} currentCoord The current coordinate being processed.
 * @param {number} coordIndex The current index of the coordinate being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed.
 * @param {number} geometryIndex The current index of the Geometry being processed.
 * @returns {void}
 */
/**
 * Iterate over coordinates in any GeoJSON object, similar to Array.forEach()
 *
 * @function
 * @param {AllGeoJSON} geojson any GeoJSON object
 * @param {coordEachCallback} callback a method that takes (currentCoord, coordIndex, featureIndex, multiFeatureIndex)
 * @param {boolean} [excludeWrapCoord=false] whether or not to include the final coordinate of LinearRings that wraps the ring in its iteration.
 * @returns {void}
 * @example
 * var features = turf.featureCollection([
 *   turf.point([26, 37], {"foo": "bar"}),
 *   turf.point([36, 53], {"hello": "world"})
 * ]);
 *
 * turf.coordEach(features, function (currentCoord, coordIndex, featureIndex, multiFeatureIndex, geometryIndex) {
 *   //=currentCoord
 *   //=coordIndex
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   //=geometryIndex
 * });
 */
declare function coordEach(geojson: AllGeoJSON, callback: (currentCoord: number[], coordIndex: number, featureIndex: number, multiFeatureIndex: number, geometryIndex: number) => void, excludeWrapCoord?: boolean): void;
/**
 * Callback for coordReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback coordReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {number[]} currentCoord The current coordinate being processed.
 * @param {number} coordIndex The current index of the coordinate being processed.
 * Starts at index 0, if an initialValue is provided, and at index 1 otherwise.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed.
 * @param {number} geometryIndex The current index of the Geometry being processed.
 * @returns {Reducer}
 */
/**
 * Reduce coordinates in any GeoJSON object, similar to Array.reduce()
 *
 * @function
 * @param {AllGeoJSON} geojson any GeoJSON object
 * @param {coordReduceCallback} callback a method that takes (previousValue, currentCoord, coordIndex)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @param {boolean} [excludeWrapCoord=false] whether or not to include the final coordinate of LinearRings that wraps the ring in its iteration.
 * @returns {Reducer} The value that results from the reduction.
 * @example
 * var features = turf.featureCollection([
 *   turf.point([26, 37], {"foo": "bar"}),
 *   turf.point([36, 53], {"hello": "world"})
 * ]);
 *
 * turf.coordReduce(features, function (previousValue, currentCoord, coordIndex, featureIndex, multiFeatureIndex, geometryIndex) {
 *   //=previousValue
 *   //=currentCoord
 *   //=coordIndex
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   //=geometryIndex
 *   return currentCoord;
 * });
 */
declare function coordReduce<Reducer>(geojson: AllGeoJSON, callback: (previousValue: Reducer, currentCoord: number[], coordIndex: number, featureIndex: number, multiFeatureIndex: number, geometryIndex: number) => Reducer, initialValue?: Reducer, excludeWrapCoord?: boolean): Reducer;
/**
 * Callback for propEach
 *
 * @callback propEachCallback
 * @param {GeoJsonProperties} currentProperties The current Properties being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @returns {void}
 */
/**
 * Iterate over properties in any GeoJSON object, similar to Array.forEach()
 *
 * @function
 * @param {FeatureCollection|Feature} geojson any GeoJSON object
 * @param {propEachCallback} callback a method that takes (currentProperties, featureIndex)
 * @returns {void}
 * @example
 * var features = turf.featureCollection([
 *     turf.point([26, 37], {foo: 'bar'}),
 *     turf.point([36, 53], {hello: 'world'})
 * ]);
 *
 * turf.propEach(features, function (currentProperties, featureIndex) {
 *   //=currentProperties
 *   //=featureIndex
 * });
 */
declare function propEach<Props extends GeoJsonProperties>(geojson: Feature<any> | FeatureCollection<any> | Feature<GeometryCollection>, callback: (currentProperties: Props, featureIndex: number) => void): void;
/**
 * Callback for propReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback propReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {GeoJsonProperties} currentProperties The current Properties being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @returns {Reducer}
 */
/**
 * Reduce properties in any GeoJSON object into a single value,
 * similar to how Array.reduce works. However, in this case we lazily run
 * the reduction, so an array of all properties is unnecessary.
 *
 * @function
 * @param {FeatureCollection|Feature|Geometry} geojson any GeoJSON object
 * @param {propReduceCallback} callback a method that takes (previousValue, currentProperties, featureIndex)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @returns {Reducer} The value that results from the reduction.
 * @example
 * var features = turf.featureCollection([
 *     turf.point([26, 37], {foo: 'bar'}),
 *     turf.point([36, 53], {hello: 'world'})
 * ]);
 *
 * turf.propReduce(features, function (previousValue, currentProperties, featureIndex) {
 *   //=previousValue
 *   //=currentProperties
 *   //=featureIndex
 *   return currentProperties
 * });
 */
declare function propReduce<Reducer, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<any, P> | FeatureCollection<any, P> | Geometry, callback: (previousValue: Reducer, currentProperties: P, featureIndex: number) => Reducer, initialValue?: Reducer): Reducer;
/**
 * Callback for featureEach
 *
 * @callback featureEachCallback
 * @param {Feature<any>} currentFeature The current Feature being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @returns {void}
 */
/**
 * Iterate over features in any GeoJSON object, similar to
 * Array.forEach.
 *
 * @function
 * @param {FeatureCollection|Feature|Feature<GeometryCollection>} geojson any GeoJSON object
 * @param {featureEachCallback} callback a method that takes (currentFeature, featureIndex)
 * @returns {void}
 * @example
 * var features = turf.featureCollection([
 *   turf.point([26, 37], {foo: 'bar'}),
 *   turf.point([36, 53], {hello: 'world'})
 * ]);
 *
 * turf.featureEach(features, function (currentFeature, featureIndex) {
 *   //=currentFeature
 *   //=featureIndex
 * });
 */
declare function featureEach<G extends GeometryObject, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | Feature<GeometryCollection, P>, callback: (currentFeature: Feature<G, P>, featureIndex: number) => void): void;
/**
 * Callback for featureReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback featureReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {Feature} currentFeature The current Feature being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @returns {Reducer}
 */
/**
 * Reduce features in any GeoJSON object, similar to Array.reduce().
 *
 * @function
 * @param {FeatureCollection|Feature|Feature<GeometryCollection>} geojson any GeoJSON object
 * @param {featureReduceCallback} callback a method that takes (previousValue, currentFeature, featureIndex)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @returns {Reducer} The value that results from the reduction.
 * @example
 * var features = turf.featureCollection([
 *   turf.point([26, 37], {"foo": "bar"}),
 *   turf.point([36, 53], {"hello": "world"})
 * ]);
 *
 * turf.featureReduce(features, function (previousValue, currentFeature, featureIndex) {
 *   //=previousValue
 *   //=currentFeature
 *   //=featureIndex
 *   return currentFeature
 * });
 */
declare function featureReduce<Reducer, G extends GeometryObject, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | Feature<GeometryCollection, P>, callback: (previousValue: Reducer, currentFeature: Feature<G, P>, featureIndex: number) => Reducer, initialValue?: Reducer): Reducer;
/**
 * Get all coordinates from any GeoJSON object.
 *
 * @function
 * @param {AllGeoJSON} geojson any GeoJSON object
 * @returns {Array<Array<number>>} coordinate position array
 * @example
 * var features = turf.featureCollection([
 *   turf.point([26, 37], {foo: 'bar'}),
 *   turf.point([36, 53], {hello: 'world'})
 * ]);
 *
 * var coords = turf.coordAll(features);
 * //= [[26, 37], [36, 53]]
 */
declare function coordAll(geojson: AllGeoJSON): number[][];
/**
 * Callback for geomEach
 *
 * @callback geomEachCallback
 * @param {GeometryObject} currentGeometry The current Geometry being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {GeoJsonProperties} featureProperties The current Feature Properties being processed.
 * @param {BBox} featureBBox The current Feature BBox being processed.
 * @param {Id} featureId The current Feature Id being processed.
 * @returns {void}
 */
/**
 * Iterate over each geometry in any GeoJSON object, similar to Array.forEach()
 *
 * @function
 * @param {FeatureCollection|Feature|Geometry|GeometryObject|Feature<GeometryCollection>} geojson any GeoJSON object
 * @param {geomEachCallback} callback a method that takes (currentGeometry, featureIndex, featureProperties, featureBBox, featureId)
 * @returns {void}
 * @example
 * var features = turf.featureCollection([
 *     turf.point([26, 37], {foo: 'bar'}),
 *     turf.point([36, 53], {hello: 'world'})
 * ]);
 *
 * turf.geomEach(features, function (currentGeometry, featureIndex, featureProperties, featureBBox, featureId) {
 *   //=currentGeometry
 *   //=featureIndex
 *   //=featureProperties
 *   //=featureBBox
 *   //=featureId
 * });
 */
declare function geomEach<G extends GeometryObject | null, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | G | GeometryCollection | Feature<GeometryCollection, P>, callback: (currentGeometry: G, featureIndex: number, featureProperties: P, featureBBox: BBox, featureId: Id) => void): void;
/**
 * Callback for geomReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback geomReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {GeometryObject} currentGeometry The current Geometry being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {GeoJsonProperties} featureProperties The current Feature Properties being processed.
 * @param {BBox} featureBBox The current Feature BBox being processed.
 * @param {Id} featureId The current Feature Id being processed.
 * @returns {Reducer}
 */
/**
 * Reduce geometry in any GeoJSON object, similar to Array.reduce().
 *
 * @function
 * @param {FeatureCollection|Feature|GeometryObject|GeometryCollection|Feature<GeometryCollection>} geojson any GeoJSON object
 * @param {geomReduceCallback} callback a method that takes (previousValue, currentGeometry, featureIndex, featureProperties, featureBBox, featureId)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @returns {Reducer} The value that results from the reduction.
 * @example
 * var features = turf.featureCollection([
 *     turf.point([26, 37], {foo: 'bar'}),
 *     turf.point([36, 53], {hello: 'world'})
 * ]);
 *
 * turf.geomReduce(features, function (previousValue, currentGeometry, featureIndex, featureProperties, featureBBox, featureId) {
 *   //=previousValue
 *   //=currentGeometry
 *   //=featureIndex
 *   //=featureProperties
 *   //=featureBBox
 *   //=featureId
 *   return currentGeometry
 * });
 */
declare function geomReduce<Reducer, G extends GeometryObject, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | G | GeometryCollection | Feature<GeometryCollection, P>, callback: (previousValue: Reducer, currentGeometry: G, featureIndex: number, featureProperties: P, featureBBox: BBox, featureId: Id) => Reducer, initialValue?: Reducer): Reducer;
/**
 * Callback for flattenEach
 *
 * @callback flattenEachCallback
 * @param {Feature} currentFeature The current flattened feature being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed.
 * @returns {void}
 */
/**
 * Iterate over flattened features in any GeoJSON object, similar to
 * Array.forEach.
 *
 * @function
 * @param {FeatureCollection|Feature|GeometryObject|GeometryCollection|Feature<GeometryCollection>} geojson any GeoJSON object
 * @param {flattenEachCallback} callback a method that takes (currentFeature, featureIndex, multiFeatureIndex)
 * @returns {void}
 * @example
 * var features = turf.featureCollection([
 *     turf.point([26, 37], {foo: 'bar'}),
 *     turf.multiPoint([[40, 30], [36, 53]], {hello: 'world'})
 * ]);
 *
 * turf.flattenEach(features, function (currentFeature, featureIndex, multiFeatureIndex) {
 *   //=currentFeature
 *   //=featureIndex
 *   //=multiFeatureIndex
 * });
 */
declare function flattenEach<G extends GeometryObject = GeometryObject, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | G | GeometryCollection | Feature<GeometryCollection, P>, callback: (currentFeature: Feature<G, P>, featureIndex: number, multiFeatureIndex: number) => void): void;
/**
 * Callback for flattenReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback flattenReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {Feature} currentFeature The current Feature being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed.
 * @returns {Reducer}
 */
/**
 * Reduce flattened features in any GeoJSON object, similar to Array.reduce().
 *
 * @function
 * @param {FeatureCollection|Feature|GeometryObject|GeometryCollection|Feature<GeometryCollection>} geojson any GeoJSON object
 * @param {flattenReduceCallback} callback a method that takes (previousValue, currentFeature, featureIndex, multiFeatureIndex)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @returns {Reducer} The value that results from the reduction.
 * @example
 * var features = turf.featureCollection([
 *     turf.point([26, 37], {foo: 'bar'}),
 *     turf.multiPoint([[40, 30], [36, 53]], {hello: 'world'})
 * ]);
 *
 * turf.flattenReduce(features, function (previousValue, currentFeature, featureIndex, multiFeatureIndex) {
 *   //=previousValue
 *   //=currentFeature
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   return currentFeature
 * });
 */
declare function flattenReduce<Reducer, G extends GeometryObject, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | G | GeometryCollection | Feature<GeometryCollection, P>, callback: (previousValue: Reducer, currentFeature: Feature<G, P>, featureIndex: number, multiFeatureIndex: number) => Reducer, initialValue?: Reducer): Reducer;
/**
 * Callback for segmentEach
 *
 * @callback segmentEachCallback
 * @param {Feature<LineString>} currentSegment The current Segment being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed.
 * @param {number} geometryIndex The current index of the Geometry being processed.
 * @param {number} segmentIndex The current index of the Segment being processed.
 * @returns {void}
 */
/**
 * Iterate over 2-vertex line segment in any GeoJSON object, similar to Array.forEach()
 * (Multi)Point geometries do not contain segments therefore they are ignored during this operation.
 *
 * @param {AllGeoJSON} geojson any GeoJSON
 * @param {segmentEachCallback} callback a method that takes (currentSegment, featureIndex, multiFeatureIndex, geometryIndex, segmentIndex)
 * @returns {void}
 * @example
 * var polygon = turf.polygon([[[-50, 5], [-40, -10], [-50, -10], [-40, 5], [-50, 5]]]);
 *
 * // Iterate over GeoJSON by 2-vertex segments
 * turf.segmentEach(polygon, function (currentSegment, featureIndex, multiFeatureIndex, geometryIndex, segmentIndex) {
 *   //=currentSegment
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   //=geometryIndex
 *   //=segmentIndex
 * });
 *
 * // Calculate the total number of segments
 * var total = 0;
 * turf.segmentEach(polygon, function () {
 *     total++;
 * });
 */
declare function segmentEach<P extends GeoJsonProperties = GeoJsonProperties>(geojson: AllGeoJSON, callback: (currentSegment?: Feature<LineString, P>, featureIndex?: number, multiFeatureIndex?: number, segmentIndex?: number, geometryIndex?: number) => void): void;
/**
 * Callback for segmentReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback segmentReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {Feature<LineString>} currentSegment The current Segment being processed.
 * @param {number} featureIndex The current index of the Feature being processed.
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed.
 * @param {number} geometryIndex The current index of the Geometry being processed.
 * @param {number} segmentIndex The current index of the Segment being processed.
 * @returns {Reducer}
 */
/**
 * Reduce 2-vertex line segment in any GeoJSON object, similar to Array.reduce()
 * (Multi)Point geometries do not contain segments therefore they are ignored during this operation.
 *
 * @param {FeatureCollection|Feature|Geometry} geojson any GeoJSON
 * @param {segmentReduceCallback} callback a method that takes (previousValue, currentSegment, currentIndex)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @returns {Reducer}
 * @example
 * var polygon = turf.polygon([[[-50, 5], [-40, -10], [-50, -10], [-40, 5], [-50, 5]]]);
 *
 * // Iterate over GeoJSON by 2-vertex segments
 * turf.segmentReduce(polygon, function (previousSegment, currentSegment, featureIndex, multiFeatureIndex, geometryIndex, segmentIndex) {
 *   //= previousSegment
 *   //= currentSegment
 *   //= featureIndex
 *   //= multiFeatureIndex
 *   //= geometryIndex
 *   //= segmentIndex
 *   return currentSegment
 * });
 *
 * // Calculate the total number of segments
 * var initialValue = 0
 * var total = turf.segmentReduce(polygon, function (previousValue) {
 *     previousValue++;
 *     return previousValue;
 * }, initialValue);
 */
declare function segmentReduce<Reducer, P extends GeoJsonProperties = GeoJsonProperties>(geojson: FeatureCollection<Lines, P> | Feature<Lines, P> | Lines | Feature<GeometryCollection, P> | GeometryCollection, callback: (previousValue?: Reducer, currentSegment?: Feature<LineString, P>, featureIndex?: number, multiFeatureIndex?: number, segmentIndex?: number, geometryIndex?: number) => Reducer, initialValue?: Reducer): Reducer;
/**
 * Callback for lineEach
 *
 * @callback lineEachCallback
 * @param {Feature<LineString>} currentLine The current LineString|LinearRing being processed
 * @param {number} featureIndex The current index of the Feature being processed
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed
 * @param {number} geometryIndex The current index of the Geometry being processed
 * @returns {void}
 */
/**
 * Iterate over line or ring coordinates in LineString, Polygon, MultiLineString, MultiPolygon Features or Geometries,
 * similar to Array.forEach.
 *
 * @function
 * @param {FeatureCollection<Lines>|Feature<Lines>|Lines|Feature<GeometryCollection>|GeometryCollection} geojson object
 * @param {lineEachCallback} callback a method that takes (currentLine, featureIndex, multiFeatureIndex, geometryIndex)
 * @returns {void}
 * @example
 * var multiLine = turf.multiLineString([
 *   [[26, 37], [35, 45]],
 *   [[36, 53], [38, 50], [41, 55]]
 * ]);
 *
 * turf.lineEach(multiLine, function (currentLine, featureIndex, multiFeatureIndex, geometryIndex) {
 *   //=currentLine
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   //=geometryIndex
 * });
 */
declare function lineEach<P extends GeoJsonProperties = GeoJsonProperties>(geojson: FeatureCollection<Lines, P> | Feature<Lines, P> | Lines | Feature<GeometryCollection, P> | GeometryCollection, callback: (currentLine: Feature<LineString, P>, featureIndex?: number, multiFeatureIndex?: number, geometryIndex?: number) => void): void;
/**
 * Callback for lineReduce
 *
 * The first time the callback function is called, the values provided as arguments depend
 * on whether the reduce method has an initialValue argument.
 *
 * If an initialValue is provided to the reduce method:
 *  - The previousValue argument is initialValue.
 *  - The currentValue argument is the value of the first element present in the array.
 *
 * If an initialValue is not provided:
 *  - The previousValue argument is the value of the first element present in the array.
 *  - The currentValue argument is the value of the second element present in the array.
 *
 * @callback lineReduceCallback
 * @param {Reducer} previousValue The accumulated value previously returned in the last invocation
 * of the callback, or initialValue, if supplied.
 * @param {Feature<LineString>} currentLine The current LineString|LinearRing being processed.
 * @param {number} featureIndex The current index of the Feature being processed
 * @param {number} multiFeatureIndex The current index of the Multi-Feature being processed
 * @param {number} geometryIndex The current index of the Geometry being processed
 * @returns {Reducer}
 */
/**
 * Reduce features in any GeoJSON object, similar to Array.reduce().
 *
 * @function
 * @param {FeatureCollection<Lines>|Feature<Lines>|Lines|Feature<GeometryCollection>|GeometryCollection} geojson object
 * @param {Function} callback a method that takes (previousValue, currentLine, featureIndex, multiFeatureIndex, geometryIndex)
 * @param {Reducer} [initialValue] Value to use as the first argument to the first call of the callback.
 * @returns {Reducer} The value that results from the reduction.
 * @example
 * var multiPoly = turf.multiPolygon([
 *   turf.polygon([[[12,48],[2,41],[24,38],[12,48]], [[9,44],[13,41],[13,45],[9,44]]]),
 *   turf.polygon([[[5, 5], [0, 0], [2, 2], [4, 4], [5, 5]]])
 * ]);
 *
 * turf.lineReduce(multiPoly, function (previousValue, currentLine, featureIndex, multiFeatureIndex, geometryIndex) {
 *   //=previousValue
 *   //=currentLine
 *   //=featureIndex
 *   //=multiFeatureIndex
 *   //=geometryIndex
 *   return currentLine
 * });
 */
declare function lineReduce<Reducer, P extends GeoJsonProperties = GeoJsonProperties>(geojson: FeatureCollection<Lines, P> | Feature<Lines, P> | Lines | Feature<GeometryCollection, P> | GeometryCollection, callback: (previousValue?: Reducer, currentLine?: Feature<LineString, P>, featureIndex?: number, multiFeatureIndex?: number, geometryIndex?: number) => Reducer, initialValue?: Reducer): Reducer;
/**
 * Finds a particular 2-vertex LineString Segment from a GeoJSON using `@turf/meta` indexes.
 *
 * Negative indexes are permitted.
 * Point & MultiPoint will always return null.
 *
 * @param {FeatureCollection|Feature|Geometry} geojson Any GeoJSON Feature or Geometry
 * @param {Object} [options={}] Optional parameters
 * @param {number} [options.featureIndex=0] Feature Index
 * @param {number} [options.multiFeatureIndex=0] Multi-Feature Index
 * @param {number} [options.geometryIndex=0] Geometry Index
 * @param {number} [options.segmentIndex=0] Segment Index
 * @param {Object} [options.properties={}] Translate Properties to output LineString
 * @param {BBox} [options.bbox={}] Translate BBox to output LineString
 * @param {number|string} [options.id={}] Translate Id to output LineString
 * @returns {Feature<LineString>} 2-vertex GeoJSON Feature LineString
 * @example
 * var multiLine = turf.multiLineString([
 *     [[10, 10], [50, 30], [30, 40]],
 *     [[-10, -10], [-50, -30], [-30, -40]]
 * ]);
 *
 * // First Segment (defaults are 0)
 * turf.findSegment(multiLine);
 * // => Feature<LineString<[[10, 10], [50, 30]]>>
 *
 * // First Segment of 2nd Multi Feature
 * turf.findSegment(multiLine, {multiFeatureIndex: 1});
 * // => Feature<LineString<[[-10, -10], [-50, -30]]>>
 *
 * // Last Segment of Last Multi Feature
 * turf.findSegment(multiLine, {multiFeatureIndex: -1, segmentIndex: -1});
 * // => Feature<LineString<[[-50, -30], [-30, -40]]>>
 */
declare function findSegment<G extends LineString | MultiLineString | Polygon | MultiPolygon, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | G, options?: {
    featureIndex?: number;
    multiFeatureIndex?: number;
    geometryIndex?: number;
    segmentIndex?: number;
    properties?: P;
    bbox?: BBox;
    id?: Id;
}): Feature<LineString, P>;
/**
 * Finds a particular Point from a GeoJSON using `@turf/meta` indexes.
 *
 * Negative indexes are permitted.
 *
 * @param {FeatureCollection|Feature|Geometry} geojson Any GeoJSON Feature or Geometry
 * @param {Object} [options={}] Optional parameters
 * @param {number} [options.featureIndex=0] Feature Index
 * @param {number} [options.multiFeatureIndex=0] Multi-Feature Index
 * @param {number} [options.geometryIndex=0] Geometry Index
 * @param {number} [options.coordIndex=0] Coord Index
 * @param {Object} [options.properties={}] Translate Properties to output Point
 * @param {BBox} [options.bbox={}] Translate BBox to output Point
 * @param {number|string} [options.id={}] Translate Id to output Point
 * @returns {Feature<Point>} 2-vertex GeoJSON Feature Point
 * @example
 * var multiLine = turf.multiLineString([
 *     [[10, 10], [50, 30], [30, 40]],
 *     [[-10, -10], [-50, -30], [-30, -40]]
 * ]);
 *
 * // First Segment (defaults are 0)
 * turf.findPoint(multiLine);
 * // => Feature<Point<[10, 10]>>
 *
 * // First Segment of the 2nd Multi-Feature
 * turf.findPoint(multiLine, {multiFeatureIndex: 1});
 * // => Feature<Point<[-10, -10]>>
 *
 * // Last Segment of last Multi-Feature
 * turf.findPoint(multiLine, {multiFeatureIndex: -1, coordIndex: -1});
 * // => Feature<Point<[-30, -40]>>
 */
declare function findPoint<G extends GeometryObject, P extends GeoJsonProperties = GeoJsonProperties>(geojson: Feature<G, P> | FeatureCollection<G, P> | G, options?: {
    featureIndex?: number;
    multiFeatureIndex?: number;
    geometryIndex?: number;
    coordIndex?: number;
    properties?: P;
    bbox?: BBox;
    id?: Id;
}): Feature<Point, P>;

export { coordAll, coordEach, coordReduce, featureEach, featureReduce, findPoint, findSegment, flattenEach, flattenReduce, geomEach, geomReduce, lineEach, lineReduce, propEach, propReduce, segmentEach, segmentReduce };
