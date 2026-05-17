"""SimulationConfig Pydantic v2 model for the Synthoseis simulation launcher."""
from __future__ import annotations

import json
import pathlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SandFraction(BaseModel):
    min: float = Field(ge=0, le=1)
    max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def min_lt_max(self) -> "SandFraction":
        if self.min >= self.max:
            raise ValueError("sand_layer_fraction min must be less than max")
        return self


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    project_folder: str
    work_folder: str
    run_id: str | None = None

    cube_shape: list[int] = Field(min_length=3, max_length=3)
    incident_angles: list[int] = Field(min_length=1, max_length=5)
    digi: int = Field(gt=0)
    infill_factor: int = Field(gt=0)
    initial_layer_stdev: list[float] = Field(min_length=2, max_length=2)
    thickness_min: int = Field(gt=0)
    thickness_max: int = Field(gt=0)
    seabed_min_depth: list[int] = Field(min_length=2, max_length=2)
    signal_to_noise_ratio_db: list[float] = Field(min_length=3, max_length=3)
    bandwidth_low: list[float] = Field(min_length=2, max_length=2)
    bandwidth_high: list[float] = Field(min_length=2, max_length=2)
    bandwidth_ord: int = Field(gt=0)
    dip_factor_max: float = Field(ge=0)
    min_number_faults: int = Field(ge=0)
    max_number_faults: int = Field(ge=0)
    pad_samples: int = Field(ge=0)
    max_column_height: list[float] = Field(min_length=2, max_length=2)
    closure_types: list[Literal["simple", "faulted", "onlap"]] = Field(min_length=1)
    min_closure_voxels_simple: int = Field(gt=0)
    min_closure_voxels_faulted: int = Field(gt=0)
    min_closure_voxels_onlap: int = Field(gt=0)
    sand_layer_thickness: int = Field(gt=0)
    sand_layer_fraction: SandFraction

    extra_qc_plots: bool = True
    verbose: bool = True
    partial_voxels: bool = True
    variable_shale_ng: bool = False
    basin_floor_fans: bool = False
    include_channels: bool = False
    include_salt: bool = True
    write_to_hdf: bool = False
    broadband_qc_volume: bool = False
    model_qc_volumes: bool = True
    multiprocess_bp: bool = True
    model_store_in_memory: bool = False
    cleanup_intermediates: bool = True

    @model_validator(mode="after")
    def thickness_ordering(self) -> "SimulationConfig":
        if self.thickness_min >= self.thickness_max:
            raise ValueError(
                "thickness_min must be strictly less than thickness_max"
            )
        return self

    @model_validator(mode="after")
    def faults_ordering(self) -> "SimulationConfig":
        if self.min_number_faults > self.max_number_faults:
            raise ValueError(
                "min_number_faults must be less than or equal to max_number_faults"
            )
        return self

    @field_validator("cube_shape")
    @classmethod
    def positive_dimensions(cls, v: list[int]) -> list[int]:
        if any(dim <= 0 for dim in v):
            raise ValueError("All cube_shape dimensions must be positive integers")
        return v

    def to_config_json(self) -> dict:
        """Serialise to the exact dict structure Parameters.__init__ expects."""
        d = self.model_dump(exclude={"run_id"})
        d["sand_layer_fraction"] = {
            "min": self.sand_layer_fraction.min,
            "max": self.sand_layer_fraction.max,
        }
        return d
