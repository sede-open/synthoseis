import { type IconName } from "./iconNames";
import { type IconPaths, IconSize } from "./iconTypes";
/** Given an icon name and size, loads the icon paths that define it. */
export type IconPathsLoader = (iconName: IconName, iconSize: IconSize) => Promise<IconPaths>;
export interface IconLoaderOptions {
    /**
     * The id of a built-in loader, or a custom loader function.
     *
     * @see https://blueprintjs.com/docs/versions/5/#icons/loading-icons
     * @default undefined (equivalent to "split-by-size")
     */
    loader?: "split-by-size" | "all" | IconPathsLoader;
}
/**
 * Blueprint icons loader.
 */
export declare class Icons {
    /**
     * Set global icon loading options for all subsequent `Icons.load()` calls.
     */
    static setLoaderOptions(options: IconLoaderOptions): void;
    /**
     * Load a single icon for use in Blueprint components.
     */
    static load(icon: IconName, size: IconSize, options?: IconLoaderOptions): Promise<void>;
    /**
     * Load a set of icons for use in Blueprint components.
     */
    static load(icons: IconName[], size: number, options?: IconLoaderOptions): Promise<void>;
    /**
     * Load all available icons for use in Blueprint components.
     */
    static loadAll(options?: IconLoaderOptions): Promise<void>;
    /**
     * Get the icon SVG paths. Returns `undefined` if the icon has not been loaded yet.
     */
    static getPaths(icon: IconName, size: IconSize): IconPaths | undefined;
    private static loadImpl;
    /**
     * @returns true if the given string is a valid {@link IconName}
     */
    static isValidIconName(iconName: string): iconName is IconName;
}
