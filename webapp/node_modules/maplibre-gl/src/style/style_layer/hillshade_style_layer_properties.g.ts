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


export type HillshadePaintProps = {
    "hillshade-illumination-direction": DataConstantProperty<number>,
    "hillshade-illumination-anchor": DataConstantProperty<"map" | "viewport">,
    "hillshade-exaggeration": DataConstantProperty<number>,
    "hillshade-shadow-color": DataConstantProperty<Color>,
    "hillshade-highlight-color": DataConstantProperty<Color>,
    "hillshade-accent-color": DataConstantProperty<Color>,
};

export type HillshadePaintPropsPossiblyEvaluated = {
    "hillshade-illumination-direction": number,
    "hillshade-illumination-anchor": "map" | "viewport",
    "hillshade-exaggeration": number,
    "hillshade-shadow-color": Color,
    "hillshade-highlight-color": Color,
    "hillshade-accent-color": Color,
};

let paint: Properties<HillshadePaintProps>;
const getPaint = () => paint = paint || new Properties({
    "hillshade-illumination-direction": new DataConstantProperty(styleSpec["paint_hillshade"]["hillshade-illumination-direction"] as any as StylePropertySpecification),
    "hillshade-illumination-anchor": new DataConstantProperty(styleSpec["paint_hillshade"]["hillshade-illumination-anchor"] as any as StylePropertySpecification),
    "hillshade-exaggeration": new DataConstantProperty(styleSpec["paint_hillshade"]["hillshade-exaggeration"] as any as StylePropertySpecification),
    "hillshade-shadow-color": new DataConstantProperty(styleSpec["paint_hillshade"]["hillshade-shadow-color"] as any as StylePropertySpecification),
    "hillshade-highlight-color": new DataConstantProperty(styleSpec["paint_hillshade"]["hillshade-highlight-color"] as any as StylePropertySpecification),
    "hillshade-accent-color": new DataConstantProperty(styleSpec["paint_hillshade"]["hillshade-accent-color"] as any as StylePropertySpecification),
});

export default ({ get paint() { return getPaint() } });