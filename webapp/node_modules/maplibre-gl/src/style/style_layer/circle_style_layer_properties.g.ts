// This file is generated. Edit build/generate-style-code.ts, then run 'npm run codegen'.
/* eslint-disable */

import {latest as styleSpec} from '@maplibre/maplibre-gl-style-spec';

import {
    Properties,
    DataConstantProperty,
    DataDrivenProperty,
    CrossFadedDataDrivenProperty,
    CrossFadedProperty,
    ColorRampProperty,
    PossiblyEvaluatedPropertyValue,
    CrossFaded
} from '../properties';

import type {Color, Formatted, Padding, ResolvedImage, VariableAnchorOffsetCollection} from '@maplibre/maplibre-gl-style-spec';
import {StylePropertySpecification} from '@maplibre/maplibre-gl-style-spec';

export type CircleLayoutProps = {
    "circle-sort-key": DataDrivenProperty<number>,
};

export type CircleLayoutPropsPossiblyEvaluated = {
    "circle-sort-key": PossiblyEvaluatedPropertyValue<number>,
};

let layout: Properties<CircleLayoutProps>;
const getLayout = () => layout = layout || new Properties({
    "circle-sort-key": new DataDrivenProperty(styleSpec["layout_circle"]["circle-sort-key"] as any as StylePropertySpecification),
});

export type CirclePaintProps = {
    "circle-radius": DataDrivenProperty<number>,
    "circle-color": DataDrivenProperty<Color>,
    "circle-blur": DataDrivenProperty<number>,
    "circle-opacity": DataDrivenProperty<number>,
    "circle-translate": DataConstantProperty<[number, number]>,
    "circle-translate-anchor": DataConstantProperty<"map" | "viewport">,
    "circle-pitch-scale": DataConstantProperty<"map" | "viewport">,
    "circle-pitch-alignment": DataConstantProperty<"map" | "viewport">,
    "circle-stroke-width": DataDrivenProperty<number>,
    "circle-stroke-color": DataDrivenProperty<Color>,
    "circle-stroke-opacity": DataDrivenProperty<number>,
};

export type CirclePaintPropsPossiblyEvaluated = {
    "circle-radius": PossiblyEvaluatedPropertyValue<number>,
    "circle-color": PossiblyEvaluatedPropertyValue<Color>,
    "circle-blur": PossiblyEvaluatedPropertyValue<number>,
    "circle-opacity": PossiblyEvaluatedPropertyValue<number>,
    "circle-translate": [number, number],
    "circle-translate-anchor": "map" | "viewport",
    "circle-pitch-scale": "map" | "viewport",
    "circle-pitch-alignment": "map" | "viewport",
    "circle-stroke-width": PossiblyEvaluatedPropertyValue<number>,
    "circle-stroke-color": PossiblyEvaluatedPropertyValue<Color>,
    "circle-stroke-opacity": PossiblyEvaluatedPropertyValue<number>,
};

let paint: Properties<CirclePaintProps>;
const getPaint = () => paint = paint || new Properties({
    "circle-radius": new DataDrivenProperty(styleSpec["paint_circle"]["circle-radius"] as any as StylePropertySpecification),
    "circle-color": new DataDrivenProperty(styleSpec["paint_circle"]["circle-color"] as any as StylePropertySpecification),
    "circle-blur": new DataDrivenProperty(styleSpec["paint_circle"]["circle-blur"] as any as StylePropertySpecification),
    "circle-opacity": new DataDrivenProperty(styleSpec["paint_circle"]["circle-opacity"] as any as StylePropertySpecification),
    "circle-translate": new DataConstantProperty(styleSpec["paint_circle"]["circle-translate"] as any as StylePropertySpecification),
    "circle-translate-anchor": new DataConstantProperty(styleSpec["paint_circle"]["circle-translate-anchor"] as any as StylePropertySpecification),
    "circle-pitch-scale": new DataConstantProperty(styleSpec["paint_circle"]["circle-pitch-scale"] as any as StylePropertySpecification),
    "circle-pitch-alignment": new DataConstantProperty(styleSpec["paint_circle"]["circle-pitch-alignment"] as any as StylePropertySpecification),
    "circle-stroke-width": new DataDrivenProperty(styleSpec["paint_circle"]["circle-stroke-width"] as any as StylePropertySpecification),
    "circle-stroke-color": new DataDrivenProperty(styleSpec["paint_circle"]["circle-stroke-color"] as any as StylePropertySpecification),
    "circle-stroke-opacity": new DataDrivenProperty(styleSpec["paint_circle"]["circle-stroke-opacity"] as any as StylePropertySpecification),
});

export default ({ get paint() { return getPaint() }, get layout() { return getLayout() } });