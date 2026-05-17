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


export type RasterPaintProps = {
    "raster-opacity": DataConstantProperty<number>,
    "raster-hue-rotate": DataConstantProperty<number>,
    "raster-brightness-min": DataConstantProperty<number>,
    "raster-brightness-max": DataConstantProperty<number>,
    "raster-saturation": DataConstantProperty<number>,
    "raster-contrast": DataConstantProperty<number>,
    "raster-resampling": DataConstantProperty<"linear" | "nearest">,
    "raster-fade-duration": DataConstantProperty<number>,
};

export type RasterPaintPropsPossiblyEvaluated = {
    "raster-opacity": number,
    "raster-hue-rotate": number,
    "raster-brightness-min": number,
    "raster-brightness-max": number,
    "raster-saturation": number,
    "raster-contrast": number,
    "raster-resampling": "linear" | "nearest",
    "raster-fade-duration": number,
};

let paint: Properties<RasterPaintProps>;
const getPaint = () => paint = paint || new Properties({
    "raster-opacity": new DataConstantProperty(styleSpec["paint_raster"]["raster-opacity"] as any as StylePropertySpecification),
    "raster-hue-rotate": new DataConstantProperty(styleSpec["paint_raster"]["raster-hue-rotate"] as any as StylePropertySpecification),
    "raster-brightness-min": new DataConstantProperty(styleSpec["paint_raster"]["raster-brightness-min"] as any as StylePropertySpecification),
    "raster-brightness-max": new DataConstantProperty(styleSpec["paint_raster"]["raster-brightness-max"] as any as StylePropertySpecification),
    "raster-saturation": new DataConstantProperty(styleSpec["paint_raster"]["raster-saturation"] as any as StylePropertySpecification),
    "raster-contrast": new DataConstantProperty(styleSpec["paint_raster"]["raster-contrast"] as any as StylePropertySpecification),
    "raster-resampling": new DataConstantProperty(styleSpec["paint_raster"]["raster-resampling"] as any as StylePropertySpecification),
    "raster-fade-duration": new DataConstantProperty(styleSpec["paint_raster"]["raster-fade-duration"] as any as StylePropertySpecification),
});

export default ({ get paint() { return getPaint() } });