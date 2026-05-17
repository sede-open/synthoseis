/** Central configuration for the Synthoseis webapp.
 *
 * MANIFEST_URL: path/URL to the manifest.json file.
 * ZARR_BASE_URL: base URL prepended to store_path values from the manifest.
 *
 * For local development with `python -m http.server`:
 *   place manifest.json and the run folders under dist/
 *   and serve from dist/. Both values are relative to the page origin.
 */

export const MANIFEST_URL = "./manifest.json";
export const ZARR_BASE_URL = "./";
