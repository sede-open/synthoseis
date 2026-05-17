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


export type HeatmapPaintProps = {
    "heatmap-radius": DataDrivenProperty<number>,
    "heatmap-weight": DataDrivenProperty<number>,
    "heatmap-intensity": DataConstantProperty<number>,
    "heatmap-color": ColorRampProperty,
    "heatmap-opacity": DataConstantProperty<number>,
};

export type HeatmapPaintPropsPossiblyEvaluated = {
    "heatmap-radius": PossiblyEvaluatedPropertyValue<number>,
    "heatmap-weight": PossiblyEvaluatedPropertyValue<number>,
    "heatmap-intensity": number,
    "heatmap-color": ColorRampProperty,
    "heatmap-opacity": number,
};

let paint: Properties<HeatmapPaintProps>;
const getPaint = () => paint = paint || new Properties({
    "heatmap-radius": new DataDrivenProperty(styleSpec["paint_heatmap"]["heatmap-radius"] as any as StylePropertySpecification),
    "heatmap-weight": new DataDrivenProperty(styleSpec["paint_heatmap"]["heatmap-weight"] as any as StylePropertySpecification),
    "heatmap-intensity": new DataConstantProperty(styleSpec["paint_heatmap"]["heatmap-intensity"] as any as StylePropertySpecification),
    "heatmap-color": new ColorRampProperty(styleSpec["paint_heatmap"]["heatmap-color"] as any as StylePropertySpecification),
    "heatmap-opacity": new DataConstantProperty(styleSpec["paint_heatmap"]["heatmap-opacity"] as any as StylePropertySpecification),
});

export default ({ get paint() { return getPaint() } });