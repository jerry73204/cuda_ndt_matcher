//! Dynamic map loading and caching module.
//!
//! This module manages the point cloud map used for NDT matching:
//! - Stores map tiles by ID
//! - Triggers updates when vehicle moves beyond threshold
//! - Filters map points within radius of current position
//! - Combines tiles into target cloud for NDT
//!
//! ## Autoware Compatibility
//!
//! This implementation uses `GetDifferentialPointCloudMap` service for differential
//! tile loading, matching Autoware's `MapUpdateModule` behavior:
//!
//! | Feature | Autoware | This Implementation |
//! |---------|----------|---------------------|
//! | Map Loading | GetDifferentialPointCloudMap service | Same ✓ |
//! | Tile Management | Per-tile add/remove via service | Same ✓ |
//! | Update Trigger | Timer callback | Position-based check on each alignment |
//! | Secondary NDT | Yes (non-blocking updates) | No (direct update with lock) |

use crate::params::DynamicMapParams;
use crate::pointcloud;
use autoware_map_msgs::msg::{AreaInfo, PointCloudMapCellWithID};
use autoware_map_msgs::srv::{GetDifferentialPointCloudMap, GetDifferentialPointCloudMap_Request};
use geometry_msgs::msg::Point;
use parking_lot::RwLock;
use rclrs::{log_debug, log_error, log_info, log_warn, Client, Node};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const LOGGER_NAME: &str = "ndt_scan_matcher.map_module";

/// A map tile with its point cloud data
#[derive(Debug, Clone)]
pub struct MapTile {
    /// Unique identifier for this tile
    pub id: String,
    /// Center position of the tile
    pub center: Point,
    /// Point cloud data as [x, y, z] points
    pub points: Vec<[f32; 3]>,
}

/// Result of a map update check
#[derive(Debug, Clone)]
pub struct MapUpdateResult {
    /// Whether the map was updated
    pub updated: bool,
    /// Number of tiles currently loaded
    pub tiles_loaded: usize,
    /// Total points in combined map
    pub total_points: usize,
    /// Distance from last update position
    pub distance_from_last_update: f64,
    /// Time taken for map update (if updated)
    pub update_time_ms: f64,
}

/// Statistics about the current map state
#[derive(Debug, Clone, Default)]
pub struct MapStats {
    /// Number of loaded tiles
    pub tiles_loaded: usize,
    /// Total points in full map (before filtering)
    pub total_points_full: usize,
    /// Points in filtered/cached map
    pub total_points_filtered: usize,
    /// Last update position
    pub last_update_position: Option<Point>,
    /// Whether map needs rebuild
    pub needs_rebuild: bool,
}

/// Manages dynamic map loading and caching
pub struct MapUpdateModule {
    /// Map tiles indexed by ID
    tiles: RwLock<HashMap<String, MapTile>>,
    /// Last position where map was updated
    last_update_position: RwLock<Option<Point>>,
    /// Combined map points (filtered by radius)
    cached_map_points: RwLock<Vec<[f32; 3]>>,
    /// Parameters for map loading
    params: DynamicMapParams,
    /// Flag indicating map needs rebuild
    needs_rebuild: RwLock<bool>,
}

impl MapUpdateModule {
    /// Create a new map update module
    pub fn new(params: DynamicMapParams) -> Self {
        Self {
            tiles: RwLock::new(HashMap::new()),
            last_update_position: RwLock::new(None),
            cached_map_points: RwLock::new(Vec::new()),
            params,
            needs_rebuild: RwLock::new(true),
        }
    }

    /// Check if map update is needed based on current position
    ///
    /// Returns true if:
    /// - No previous update position exists (first update)
    /// - Distance from last update exceeds update_distance threshold
    pub fn should_update(&self, current_position: &Point) -> bool {
        let last_pos = self.last_update_position.read();

        match last_pos.as_ref() {
            None => true, // First update
            Some(last) => {
                let distance = euclidean_distance_2d(current_position, last);
                distance > self.params.update_distance
            }
        }
    }

    /// Get distance from last update position
    pub fn distance_from_last_update(&self, current_position: &Point) -> f64 {
        let last_pos = self.last_update_position.read();
        match last_pos.as_ref() {
            None => f64::INFINITY,
            Some(last) => euclidean_distance_2d(current_position, last),
        }
    }

    /// Check if position is out of current map range
    ///
    /// Returns true if the current position plus lidar radius
    /// exceeds the map radius from the last update position
    pub fn out_of_map_range(&self, current_position: &Point) -> bool {
        let last_pos = self.last_update_position.read();

        match last_pos.as_ref() {
            None => true,
            Some(last) => {
                let distance = euclidean_distance_2d(current_position, last);
                distance + self.params.lidar_radius > self.params.map_radius
            }
        }
    }

    /// Add or update a map tile
    pub fn add_tile(&self, tile: MapTile) {
        let mut tiles = self.tiles.write();
        tiles.insert(tile.id.clone(), tile);
        *self.needs_rebuild.write() = true;
    }

    /// Remove a map tile by ID
    pub fn remove_tile(&self, tile_id: &str) -> bool {
        let mut tiles = self.tiles.write();
        let removed = tiles.remove(tile_id).is_some();
        if removed {
            *self.needs_rebuild.write() = true;
        }
        removed
    }

    /// Get list of currently loaded tile IDs
    pub fn get_loaded_tile_ids(&self) -> Vec<String> {
        self.tiles.read().keys().cloned().collect()
    }

    /// Get number of loaded tiles
    pub fn tile_count(&self) -> usize {
        self.tiles.read().len()
    }

    /// Update the map based on current position
    ///
    /// This filters points within map_radius of the current position
    /// and rebuilds the cached map points.
    ///
    /// This method implements Autoware's `should_update_map` + `update_map` logic:
    /// - Checks if position has moved beyond `update_distance`
    /// - Checks if we're approaching the edge of the loaded map
    /// - Filters points within `map_radius` of current position
    ///
    /// Returns update result with statistics
    pub fn update_map(&self, current_position: &Point) -> MapUpdateResult {
        let distance = self.distance_from_last_update(current_position);
        let needs_rebuild = *self.needs_rebuild.read();

        // Check if we need to update (matches Autoware's should_update_map)
        if !needs_rebuild && !self.should_update(current_position) {
            let cached = self.cached_map_points.read();
            return MapUpdateResult {
                updated: false,
                tiles_loaded: self.tile_count(),
                total_points: cached.len(),
                distance_from_last_update: distance,
                update_time_ms: 0.0,
            };
        }

        let start_time = Instant::now();

        // Check if we're falling behind (Autoware's "dynamic map loading not keeping up" check)
        if distance + self.params.lidar_radius > self.params.map_radius {
            log_debug!(
                LOGGER_NAME,
                "Map update not keeping up: distance={:.1}m + lidar_radius={:.1}m > map_radius={:.1}m",
                distance,
                self.params.lidar_radius,
                self.params.map_radius
            );
        }

        // Rebuild map with points within radius
        let tiles = self.tiles.read();
        let mut combined_points = Vec::new();
        let radius_sq = (self.params.map_radius as f32).powi(2);

        for tile in tiles.values() {
            for point in &tile.points {
                let dx = point[0] - current_position.x as f32;
                let dy = point[1] - current_position.y as f32;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= radius_sq {
                    combined_points.push(*point);
                }
            }
        }

        let tiles_loaded = tiles.len();
        let total_points = combined_points.len();
        drop(tiles);

        // Update cached map
        *self.cached_map_points.write() = combined_points;
        *self.last_update_position.write() = Some(current_position.clone());
        *self.needs_rebuild.write() = false;

        let update_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        log_debug!(
            LOGGER_NAME,
            "Map updated: {} points within {:.0}m radius (took {:.1}ms)",
            total_points,
            self.params.map_radius,
            update_time_ms
        );

        MapUpdateResult {
            updated: true,
            tiles_loaded,
            total_points,
            distance_from_last_update: distance,
            update_time_ms,
        }
    }

    /// Get statistics about the current map state
    pub fn get_stats(&self) -> MapStats {
        let tiles = self.tiles.read();
        let total_points_full: usize = tiles.values().map(|t| t.points.len()).sum();
        let tiles_loaded = tiles.len();
        drop(tiles);

        MapStats {
            tiles_loaded,
            total_points_full,
            total_points_filtered: self.cached_map_points.read().len(),
            last_update_position: self.last_update_position.read().clone(),
            needs_rebuild: *self.needs_rebuild.read(),
        }
    }

    /// Check and update map if needed, returning whether NDT target needs updating
    ///
    /// This is a convenience method that combines `should_update` and `update_map`,
    /// returning the filtered map points if an update was performed.
    ///
    /// # Returns
    /// - `Some(points)` if map was updated and NDT target should be refreshed
    /// - `None` if no update was needed
    pub fn check_and_update(&self, current_position: &Point) -> Option<Vec<[f32; 3]>> {
        let result = self.update_map(current_position);
        if result.updated {
            self.get_map_points()
        } else {
            None
        }
    }

    /// Get the current cached map points
    ///
    /// Returns None if no map is loaded
    pub fn get_map_points(&self) -> Option<Vec<[f32; 3]>> {
        let cached = self.cached_map_points.read();
        if cached.is_empty() {
            None
        } else {
            Some(cached.clone())
        }
    }

    /// Get a reference to the cached map points for reading
    pub fn get_map_points_ref(&self) -> Arc<Vec<[f32; 3]>> {
        Arc::new(self.cached_map_points.read().clone())
    }

    /// Clear all map data
    pub fn clear(&self) {
        self.tiles.write().clear();
        self.cached_map_points.write().clear();
        *self.last_update_position.write() = None;
        *self.needs_rebuild.write() = true;
    }

    /// Load a complete map (replaces all tiles)
    ///
    /// This is a convenience method for loading a single large map
    /// without tile management.
    pub fn load_full_map(&self, points: Vec<[f32; 3]>) {
        let center = if points.is_empty() {
            Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        } else {
            // Calculate centroid
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_z = 0.0f64;
            for p in &points {
                sum_x += p[0] as f64;
                sum_y += p[1] as f64;
                sum_z += p[2] as f64;
            }
            let n = points.len() as f64;
            Point {
                x: sum_x / n,
                y: sum_y / n,
                z: sum_z / n,
            }
        };

        let tile = MapTile {
            id: "full_map".to_string(),
            center,
            points,
        };

        self.clear();
        self.add_tile(tile);
    }

    /// Get the parameters
    pub fn params(&self) -> &DynamicMapParams {
        &self.params
    }
}

/// Calculate 2D Euclidean distance between two points
fn euclidean_distance_2d(a: &Point, b: &Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

// ============================================================================
// DynamicMapLoader - Service client for differential map loading
// ============================================================================

/// Dynamic map loader using GetDifferentialPointCloudMap service.
///
/// This implements Autoware's differential map loading pattern:
/// 1. Requests map tiles around the current position via service
/// 2. Receives new tiles to add and old tile IDs to remove
/// 3. Updates the MapUpdateModule with the differential changes
///
/// The service client requires the node to spin for callbacks to work.
pub struct DynamicMapLoader {
    /// Service client for map loading
    client: Client<GetDifferentialPointCloudMap>,
    /// Map update module to populate
    map_module: Arc<MapUpdateModule>,
    /// Flag indicating a request is in flight
    request_pending: Arc<AtomicBool>,
}

impl DynamicMapLoader {
    /// Create a new dynamic map loader.
    ///
    /// # Arguments
    /// * `node` - The ROS node to create the client on
    /// * `service_name` - Name of the pcd_loader_service (typically "pcd_loader_service")
    /// * `map_module` - The map module to populate with loaded tiles
    pub fn new(
        node: &Node,
        service_name: &str,
        map_module: Arc<MapUpdateModule>,
    ) -> Result<Self, rclrs::RclrsError> {
        let client = node.create_client::<GetDifferentialPointCloudMap>(service_name)?;
        log_info!(
            LOGGER_NAME,
            "Created GetDifferentialPointCloudMap client for '{service_name}'"
        );

        Ok(Self {
            client,
            map_module,
            request_pending: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Check if the map service is available.
    pub fn service_is_ready(&self) -> bool {
        self.client.service_is_ready().unwrap_or(false)
    }

    /// Check if a request is currently pending.
    pub fn is_request_pending(&self) -> bool {
        self.request_pending.load(Ordering::SeqCst)
    }

    /// Request differential map update around a position.
    ///
    /// This is an async operation - the callback will be invoked when the
    /// response arrives. The node must spin for the callback to be processed.
    ///
    /// # Arguments
    /// * `position` - Current position to request map around
    /// * `map_radius` - Radius of area to load (typically from DynamicMapParams)
    ///
    /// # Returns
    /// * `Ok(true)` - Request was sent successfully
    /// * `Ok(false)` - Request not sent (service unavailable or request pending)
    /// * `Err(_)` - Error sending request
    pub fn request_map_update(
        &self,
        position: &Point,
        map_radius: f32,
    ) -> Result<bool, rclrs::RclrsError> {
        // Check if service is available
        if !self.service_is_ready() {
            log_warn!(
                LOGGER_NAME,
                "pcd_loader_service not available, skipping map update request"
            );
            return Ok(false);
        }

        // Check if a request is already pending
        if self.request_pending.swap(true, Ordering::SeqCst) {
            log_debug!(LOGGER_NAME, "Map update request already pending, skipping");
            return Ok(false);
        }

        // Build the request
        let request = GetDifferentialPointCloudMap_Request {
            area: AreaInfo {
                center_x: position.x as f32,
                center_y: position.y as f32,
                radius: map_radius,
            },
            cached_ids: self.map_module.get_loaded_tile_ids(),
        };

        log_debug!(
            LOGGER_NAME,
            "Requesting map at ({:.1}, {:.1}) radius={:.0}m, cached_tiles={}",
            request.area.center_x,
            request.area.center_y,
            request.area.radius,
            request.cached_ids.len()
        );

        // Clone what we need for the callback
        let map_module = Arc::clone(&self.map_module);
        let request_pending = Arc::clone(&self.request_pending);

        // Send request with callback
        let _promise = self.client.call_then(request, move |response| {
            Self::handle_response(response, &map_module);
            request_pending.store(false, Ordering::SeqCst);
        })?;

        Ok(true)
    }

    /// Handle response from the map service.
    fn handle_response(
        response: autoware_map_msgs::srv::GetDifferentialPointCloudMap_Response,
        map_module: &MapUpdateModule,
    ) {
        let start_time = Instant::now();

        // Process tiles to add
        let mut tiles_added = 0;
        let mut points_added = 0;

        for cell in &response.new_pointcloud_with_ids {
            match Self::convert_cell_to_tile(cell) {
                Ok(tile) => {
                    points_added += tile.points.len();
                    map_module.add_tile(tile);
                    tiles_added += 1;
                }
                Err(e) => {
                    log_error!(
                        LOGGER_NAME,
                        "Failed to convert map cell '{}': {e}",
                        cell.cell_id
                    );
                }
            }
        }

        // Process tiles to remove
        let mut tiles_removed = 0;
        for id in &response.ids_to_remove {
            if map_module.remove_tile(id) {
                tiles_removed += 1;
            }
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        if tiles_added > 0 || tiles_removed > 0 {
            log_info!(
                LOGGER_NAME,
                "Map update: +{tiles_added} tiles ({points_added} points), -{tiles_removed} tiles, took {elapsed_ms:.1}ms"
            );
        }
    }

    /// Convert a PointCloudMapCellWithID to a MapTile.
    fn convert_cell_to_tile(cell: &PointCloudMapCellWithID) -> Result<MapTile, anyhow::Error> {
        // Convert PointCloud2 to Vec<[f32; 3]>
        let points = pointcloud::from_pointcloud2(&cell.pointcloud)?;

        // Calculate center from point cloud
        let center = if points.is_empty() {
            Point {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        } else {
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_z = 0.0f64;
            for p in &points {
                sum_x += p[0] as f64;
                sum_y += p[1] as f64;
                sum_z += p[2] as f64;
            }
            let n = points.len() as f64;
            Point {
                x: sum_x / n,
                y: sum_y / n,
                z: sum_z / n,
            }
        };

        Ok(MapTile {
            id: cell.cell_id.clone(),
            center,
            points,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params() -> DynamicMapParams {
        DynamicMapParams {
            update_distance: 20.0,
            map_radius: 150.0,
            lidar_radius: 100.0,
        }
    }

    fn make_point(x: f64, y: f64, z: f64) -> Point {
        Point { x, y, z }
    }

    #[test]
    fn test_should_update_first_time() {
        let module = MapUpdateModule::new(make_params());
        let pos = make_point(0.0, 0.0, 0.0);

        // First update should always return true
        assert!(module.should_update(&pos));
    }

    #[test]
    fn test_should_update_within_threshold() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);
        let pos2 = make_point(10.0, 0.0, 0.0); // 10m away, below 20m threshold

        module.update_map(&pos1);
        assert!(!module.should_update(&pos2));
    }

    #[test]
    fn test_should_update_beyond_threshold() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);
        let pos2 = make_point(25.0, 0.0, 0.0); // 25m away, above 20m threshold

        module.update_map(&pos1);
        assert!(module.should_update(&pos2));
    }

    #[test]
    fn test_out_of_map_range() {
        let module = MapUpdateModule::new(make_params());
        let pos1 = make_point(0.0, 0.0, 0.0);

        module.update_map(&pos1);

        // Position at 40m: 40 + 100 (lidar) = 140 < 150 (map_radius)
        let pos_safe = make_point(40.0, 0.0, 0.0);
        assert!(!module.out_of_map_range(&pos_safe));

        // Position at 60m: 60 + 100 (lidar) = 160 > 150 (map_radius)
        let pos_out = make_point(60.0, 0.0, 0.0);
        assert!(module.out_of_map_range(&pos_out));
    }

    #[test]
    fn test_add_remove_tiles() {
        let module = MapUpdateModule::new(make_params());

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };

        module.add_tile(tile);
        assert_eq!(module.tile_count(), 1);
        assert!(module.get_loaded_tile_ids().contains(&"tile_1".to_string()));

        module.remove_tile("tile_1");
        assert_eq!(module.tile_count(), 0);
    }

    #[test]
    fn test_map_radius_filtering() {
        let mut params = make_params();
        params.map_radius = 10.0; // Small radius for testing

        let module = MapUpdateModule::new(params);

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![
                [0.0, 0.0, 0.0],  // At origin - should be included
                [5.0, 0.0, 0.0],  // 5m away - should be included
                [15.0, 0.0, 0.0], // 15m away - should be excluded
                [0.0, 20.0, 0.0], // 20m away - should be excluded
            ],
        };

        module.add_tile(tile);

        let result = module.update_map(&make_point(0.0, 0.0, 0.0));

        assert!(result.updated);
        assert_eq!(result.total_points, 2); // Only points within 10m radius
        assert!(result.update_time_ms >= 0.0);
    }

    #[test]
    fn test_check_and_update() {
        let mut params = make_params();
        params.update_distance = 5.0;

        let module = MapUpdateModule::new(params);

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };
        module.add_tile(tile);

        // First call should update
        let points = module.check_and_update(&make_point(0.0, 0.0, 0.0));
        assert!(points.is_some());

        // Second call at same position should not update
        let points = module.check_and_update(&make_point(0.0, 0.0, 0.0));
        assert!(points.is_none());

        // Call at distant position should update
        let points = module.check_and_update(&make_point(10.0, 0.0, 0.0));
        assert!(points.is_some());
    }

    #[test]
    fn test_get_stats() {
        let module = MapUpdateModule::new(make_params());

        let tile = MapTile {
            id: "tile_1".to_string(),
            center: make_point(0.0, 0.0, 0.0),
            points: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        };
        module.add_tile(tile);

        let stats = module.get_stats();
        assert_eq!(stats.tiles_loaded, 1);
        assert_eq!(stats.total_points_full, 3);
        assert_eq!(stats.total_points_filtered, 0); // Not updated yet
        assert!(stats.needs_rebuild);

        // Update map
        module.update_map(&make_point(0.0, 0.0, 0.0));

        let stats = module.get_stats();
        assert!(!stats.needs_rebuild);
        assert!(stats.total_points_filtered > 0);
    }

    #[test]
    fn test_load_full_map() {
        let module = MapUpdateModule::new(make_params());

        let points = vec![[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]];

        module.load_full_map(points.clone());

        assert_eq!(module.tile_count(), 1);
        assert!(module
            .get_loaded_tile_ids()
            .contains(&"full_map".to_string()));
    }
}
