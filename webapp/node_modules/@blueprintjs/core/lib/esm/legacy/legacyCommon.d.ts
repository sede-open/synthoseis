/// <reference types="react" />
/**
 * Represents anything that has a `name` property such as Functions.
 */
export interface Named {
    name?: string;
}
/**
 * Generic interface defining constructor types, such as classes. This is used to type the class
 * itself in meta-programming situations such as decorators.
 */
export type Constructor<T> = new (...args: any[]) => T;
export declare function getDisplayName(ComponentClass: React.ComponentType | Named): string;
