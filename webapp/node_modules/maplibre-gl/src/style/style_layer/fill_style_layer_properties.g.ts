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

export type FillLayoutProps = {
    "fill-sort-key": DataDrivenProperty<number>,
};

export type FillLayoutPropsPossiblyEvaluated = {
    "fill-sort-key": PossiblyEvaluatedPropertyValue<number>,
};

let layout: Properties<FillLayoutProps>;
const getLayout = () => layout = layout || new Properties({
    "fill-sort-key": new DataDrivenProperty(styleSpec["layout_fill"]["fill-sort-key"] as any as StylePropertySpecification),
});

export type FillPaintProps = {
    "fill-antialias": DataConstantProperty<boolean>,
    "fill-opacity": DataDrivenProperty<number>,
    "fill-color": DataDrivenProperty<Color>,
    "fill-outline-color": DataDrivenProperty<Color>,
    "fill-translate": DataConstantProperty<[number, number]>,
    "fill-translate-anchor": DataConstantProperty<"map" | "viewport">,
    "fill-pattern": CrossFadedDataDrivenProperty<ResolvedImage>,
};

export type FillPaintPropsPossiblyEvaluated = {
    "fill-antialias": boolean,
    "fill-opacity": PossiblyEvaluatedPropertyValue<number>,
    "fill-color": PossiblyEvaluatedPropertyValue<Color>,
    "fill-outline-color": PossiblyEvaluatedPropertyValue<Color>,
    "fill-translate": [number, number],
    "fill-translate-anchor": "map" | "viewport",
    "fill-pattern": PossiblyEvaluatedPropertyValue<CrossFaded<ResolvedImage>>,
};

let paint: Properties<FillPaintProps>;
const getPaint = () => paint = paint || new Properties({
    "fill-antialias": new DataConstantProperty(styleSpec["paint_fill"]["fill-antialias"] as any as StylePropertySpecification),
    "fill-opacity": new DataDrivenProperty(styleSpec["paint_fill"]["fill-opacity"] as any as StylePropertySpecification),
    "fill-color": new DataDrivenProperty(styleSpec["paint_fill"]["fill-color"] as any as StylePropertySpecification),
    "fill-outline-color": new DataDrivenProperty(styleSpec["paint_fill"]["fill-outline-color"] as any as StylePropertySpecification),
    "fill-translate": new DataConstantProperty(styleSpec["paint_fill"]["fill-translate"] as any as StylePropertySpecification),
    "fill-translate-anchor": new DataConstantProperty(styleSpec["paint_fill"]["fill-translate-anchor"] as any as StylePropertySpecification),
    "fill-pattern": new CrossFadedDataDrivenProperty(styleSpec["paint_fill"]["fill-pattern"] as any as StylePropertySpecification),
});

export default ({ get paint() { return getPaint() }, get layout() { return getLayout() } });