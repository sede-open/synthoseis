/*
 * Copyright 2021 Palantir Technologies, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { IconNames, IconNamesSet } from "./iconNames";
import { IconSize } from "./iconTypes";
import { wrapWithTimer } from "./loaderUtils";
async function getLoaderFn(options) {
    const { loader = singleton.defaultLoader } = options;
    if (typeof loader === "function") {
        return loader;
    }
    else if (loader === "all") {
        return (await import(
        /* webpackChunkName: "blueprint-icons-all-paths-loader" */
        "./paths-loaders/allPathsLoader")).allPathsLoader;
    }
    else {
        return (await import(
        /* webpackChunkName: "blueprint-icons-split-paths-by-size-loader" */
        "./paths-loaders/splitPathsBySizeLoader")).splitPathsBySizeLoader;
    }
}
/**
 * Blueprint icons loader.
 */
export class Icons {
    /** @internal */
    defaultLoader = "split-by-size";
    /** @internal */
    loadedIconPaths16 = new Map();
    /** @internal */
    loadedIconPaths20 = new Map();
    /**
     * Set global icon loading options for all subsequent `Icons.load()` calls.
     */
    static setLoaderOptions(options) {
        if (options.loader !== undefined) {
            singleton.defaultLoader = options.loader;
        }
    }
    static async load(icons, size, options) {
        if (!Array.isArray(icons)) {
            icons = [icons];
        }
        await Promise.all(icons.map(icon => this.loadImpl(icon, size, options)));
        return;
    }
    /**
     * Load all available icons for use in Blueprint components.
     */
    static async loadAll(options) {
        const allIcons = Object.values(IconNames);
        wrapWithTimer(`[Blueprint] loading all icons`, async () => {
            await Promise.all([
                this.load(allIcons, IconSize.STANDARD, options),
                this.load(allIcons, IconSize.LARGE, options),
            ]);
        });
    }
    /**
     * Get the icon SVG paths. Returns `undefined` if the icon has not been loaded yet.
     */
    static getPaths(icon, size) {
        if (!this.isValidIconName(icon)) {
            // don't warn, since this.load() will have warned already
            return undefined;
        }
        const loadedIcons = size < IconSize.LARGE ? singleton.loadedIconPaths16 : singleton.loadedIconPaths20;
        return loadedIcons.get(icon);
    }
    static async loadImpl(icon, size, options = {}) {
        if (!this.isValidIconName(icon)) {
            console.error(`[Blueprint] Unknown icon '${icon}'`);
            return;
        }
        const loadedIcons = size < IconSize.LARGE ? singleton.loadedIconPaths16 : singleton.loadedIconPaths20;
        if (loadedIcons.has(icon)) {
            // already loaded, no-op
            return;
        }
        const loaderFn = await getLoaderFn(options);
        try {
            const supportedSize = size < IconSize.LARGE ? IconSize.STANDARD : IconSize.LARGE;
            const paths = await loaderFn(icon, supportedSize);
            loadedIcons.set(icon, paths);
        }
        catch (e) {
            console.error(`[Blueprint] Unable to load ${size}px icon '${icon}'`, e);
        }
    }
    /**
     * @returns true if the given string is a valid {@link IconName}
     */
    static isValidIconName(iconName) {
        return IconNamesSet.has(iconName);
    }
}
const singleton = new Icons();
//# sourceMappingURL=iconLoader.js.map