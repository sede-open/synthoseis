import os
import sys
import numpy as np
from bruges.rockphysics import moduli


class RockProperties:
    """Rock Properties for a single rock type, containing Rho, Vp, Vs"""

    def __init__(self, rho: float, vp: float, vs: float, depth: float = None):
        """This is the class that describes the rock properties for a single rock type.

        Parameters
        ----------
        rho : float
            The density of the rock in g/cc
        vp : float
            Compressional wave velocity
        vs : float
            Rock shear wave velocity
        depth : float, optional
            The depth at which the rock lives, by default None
        """
        self.rho = rho
        self.vp = vp
        self.vs = vs
        self.ai = None
        self.si = None
        self.vpvs = None
        self.pr = None
        self.m = None
        self.mu = None
        self.k = None
        self.lam = None
        self.z = depth

    def calc_ai(self) -> float:
        """calc_ai A function that calculates camplitude impedance of the rock"""
        self.ai = self.rho * self.vp

    def calc_si(self) -> float:
        """calc_si Function that computes the shear wave impedance of the rock"""
        self.si = self.rho * self.vs

    def calc_vpvs(self) -> float:
        self.vpvs = self.vp / self.vs

    def calc_pr(self) -> float:
        if self.vpvs is None:
            self.calc_vpvs()
        self.pr = (self.vpvs**2 - 2) / (2 * self.vpvs**2 - 2)

    def calc_m(self) -> float:
        self.m = moduli.pmod(vp=self.vp, rho=self.rho)

    def calc_mu(self) -> float:
        self.mu = moduli.mu(rho=self.rho, vs=self.vs)

    def calc_k(self):
        if self.m is None:
            self.calc_m()
        if self.mu is None:
            self.calc_mu()
        self.k = moduli.bulk(pmod=self.m, mu=self.mu)

    def calc_lam(self):
        self.lam = moduli.lam(rho=self.rho, vp=self.vp, vs=self.vs)

    def calc_all_elastic_properties(self):
        self.calc_ai()
        self.calc_si()
        self.calc_vpvs()
        self.calc_pr()
        self.calc_m()
        self.calc_mu()
        self.calc_k()
        self.calc_lam()


class EndMemberMixing:
    """Mixed rock properties"""

    def __init__(self, shales, sands, net_to_gross):
        self.shales = shales
        self.sands = sands
        self.net_to_gross = net_to_gross
        self.rho = None
        self.vp = None
        self.vs = None

    @staticmethod
    def _arithmetic_mean(a0, a1, weight):
        return (a0 * (1.0 - weight)) + (a1 * weight)

    @staticmethod
    def _harmonic_mean(a0, a1, weight):
        return 1.0 / ((1.0 / a0 * (1.0 - weight)) + (1.0 / a1 * weight))

    def inverse_velocity_mixing(self):
        """Mix Shales and Sands via inverse velocity mixing

        Args:
            properties0 : Shales
            properties1 : Sands
            weight : Net to gross (% sand)
        """
        self.rho = self._arithmetic_mean(
            self.shales.rho, self.sands.rho, self.net_to_gross
        )
        self.vp = self._harmonic_mean(self.shales.vp, self.sands.vp, self.net_to_gross)
        self.vs = self._harmonic_mean(self.shales.vs, self.sands.vs, self.net_to_gross)

    def backus_moduli_mixing(self):
        """Mix Shales and Sands via Backus Moduli mixing

        Args:
            properties0 : Shales
            properties1 : Sands
            weight : Net to gross (% sand)
        """
        self.shales.calc_lam()
        self.sands.calc_lam()
        lambda_mix = self._harmonic_mean(
            self.shales.lam, self.sands.lam, self.net_to_gross
        )
        self.shales.calc_mu()
        self.sands.calc_mu()
        mu_mix = self._harmonic_mean(self.shales.mu, self.sands.mu, self.net_to_gross)
        self.rho = self._arithmetic_mean(
            self.shales.rho, self.sands.rho, self.net_to_gross
        )
        self.vp = moduli.vp(lam=lambda_mix, mu=mu_mix, rho=self.rho)
        self.vs = moduli.vs(mu=mu_mix, rho=self.rho)


def decimate_array_to_1d(cfg, in_array, xy_factor=10, z_factor=1):
    x = int(cfg.cube_shape[0] / xy_factor)
    y = int(cfg.cube_shape[1] / xy_factor)
    z = z_factor
    return np.ravel(in_array[::x, ::y, ::z])


def select_rpm(cfg):
    """Select the rock property model to use.

    User must add their own rock property model to the select_rpm function,
    and set the cfg.project variable to the name of the rock property model.
    """
    if cfg.project == "example":
        from rockphysics.rpm_example import RPMExample

        rpm = RPMExample(cfg)
    elif cfg.project == "abc":
        from rockphysics.rpm_abc import RPMABC

        print("Using the abcRPM")
        rpm = RPMABC(cfg)
    else:
        print("No rock property model defined in select_rpm")
        print("Exiting code")
        sys.exit(1)
    return rpm


def rpm_qc_plots(cfg, rpm):
    from datagenerator.util import import_matplotlib

    plt = import_matplotlib()

    # 1D trendlines
    rpm = rpm.create_1d_trends()

    # Randomised rock properties. decimate a bit
    if hasattr(cfg, "bulk_z_shift"):
        depth = (
            cfg.h5file.root.ModelData.faulted_depth[::4, ::4, ::2] + cfg.bulk_z_shift
        )
    else:
        depth = cfg.h5file.root.ModelData.faulted_depth[::4, ::4, ::2]
    lith = cfg.h5file.root.ModelData.faulted_lithology[::4, ::4, ::2]
    ng = cfg.h5file.root.ModelData.faulted_net_to_gross[::4, ::4, ::2]
    rho = cfg.h5file.root.ModelData.rho[::4, ::4, ::2]
    vp = cfg.h5file.root.ModelData.vp[::4, ::4, ::2]
    vs = cfg.h5file.root.ModelData.vs[::4, ::4, ::2]
    oil = cfg.h5file.root.ModelData.oil_closures[::4, ::4, ::2]
    gas = cfg.h5file.root.ModelData.gas_closures[::4, ::4, ::2]
    if cfg.include_salt:
        salt = cfg.h5file.root.ModelData.salt_segments[::4, ::4, ::2]
        rho_salt = rho[salt > 0]
        vp_salt = vp[salt > 0]
        vs_salt = vs[salt > 0]
        z_salt = depth[salt > 0]

    rho_shale = rho[(lith == 0) & (ng >= 0)]
    vp_shale = vp[(lith == 0) & (ng >= 0)]
    vs_shale = vs[(lith == 0) & (ng >= 0)]
    z_shale = depth[(lith == 0) & (ng >= 0)]

    rho_sand = rho[(lith > 0) & (ng > 0) & (oil == 0) & (gas == 0)]
    vp_sand = vp[(lith > 0) & (ng > 0) & (oil == 0) & (gas == 0)]
    vs_sand = vs[(lith > 0) & (ng > 0) & (oil == 0) & (gas == 0)]
    z_sand = depth[(lith > 0) & (ng > 0) & (oil == 0) & (gas == 0)]

    rho_oil_sand = rho[(ng > 0) & (oil > 0) & (gas == 0)]
    vp_oil_sand = vp[(ng > 0) & (oil > 0) & (gas == 0)]
    vs_oil_sand = vs[(ng > 0) & (oil > 0) & (gas == 0)]
    z_oil_sand = depth[(ng > 0) & (oil > 0) & (gas == 0)]

    rho_gas_sand = rho[(lith > 0) & (ng > 0) & (oil == 0) & (gas > 0)]
    vp_gas_sand = vp[(lith > 0) & (ng > 0) & (oil == 0) & (gas > 0)]
    vs_gas_sand = vs[(lith > 0) & (ng > 0) & (oil == 0) & (gas > 0)]
    z_gas_sand = depth[(lith > 0) & (ng > 0) & (oil == 0) & (gas > 0)]

    # Plot properties vs depth
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)

    # Vp
    axs[0].set_xlabel("Vp")
    axs[0].scatter(
        vp_shale[::10], z_shale[::10], c="Saddlebrown", alpha=0.1, marker="."
    )
    axs[0].scatter(vp_sand, z_sand, c="Blue", alpha=0.1, marker=".")
    if np.max(gas) > 0:
        axs[0].scatter(vp_gas_sand, z_gas_sand, c="Red", alpha=0.1, marker=".")
    if np.max(oil) > 0:
        axs[0].scatter(vp_oil_sand, z_oil_sand, c="Green", alpha=0.1, marker=".")
    if cfg.include_salt:
        axs[0].scatter(vp_salt, z_salt, c="Gray", alpha=0.1, marker=".", label="Salt")
    # Add 1D trendlines
    axs[0].plot(rpm["shale_vp"], rpm["z"], label="Shale", c="Black", lw=2)
    axs[0].plot(
        rpm["brine_sand_vp"], rpm["z"], label="Brine Sand", c="DodgerBlue", lw=2
    )
    axs[0].plot(rpm["oil_sand_vp"], rpm["z"], label="Oil Sand", c="Seagreen", lw=2)
    axs[0].plot(rpm["gas_sand_vp"], rpm["z"], label="Gas Sand", c="Tomato", lw=2)

    # Vs
    axs[1].set_xlabel("Vs")
    axs[1].scatter(
        vs_shale[::10], z_shale[::10], c="Saddlebrown", alpha=0.1, marker="."
    )
    axs[1].scatter(vs_sand, z_sand, c="Blue", alpha=0.1, marker=".")
    if np.max(gas) > 0:
        axs[1].scatter(vs_gas_sand, z_gas_sand, c="Red", alpha=0.1, marker=".")
    if np.max(oil) > 0:
        axs[1].scatter(vs_oil_sand, z_oil_sand, c="Green", alpha=0.1, marker=".")
    if cfg.include_salt:
        axs[1].scatter(vs_salt, z_salt, c="Gray", alpha=0.1, marker=".")
    axs[1].plot(rpm["shale_vs"], rpm["z"], label="Shale", c="Black", lw=2)
    axs[1].plot(
        rpm["brine_sand_vs"], rpm["z"], label="Brine Sand", c="DodgerBlue", lw=2
    )
    axs[1].plot(rpm["oil_sand_vs"], rpm["z"], label="Oil Sand", c="Seagreen", lw=2)
    axs[1].plot(rpm["gas_sand_vs"], rpm["z"], label="Gas Sand", c="Tomato", lw=2)

    # Rho
    axs[2].set_xlabel("Rho")
    axs[2].scatter(
        rho_shale[::10], z_shale[::10], c="Saddlebrown", alpha=0.1, marker="."
    )
    axs[2].scatter(rho_sand[::10], z_sand[::10], c="Blue", alpha=0.1, marker=".")
    if np.max(gas) > 0:
        axs[2].scatter(rho_gas_sand, z_gas_sand, c="Red", alpha=0.1, marker=".")
    if np.max(oil) > 0:
        axs[2].scatter(rho_oil_sand, z_oil_sand, c="Green", alpha=0.1, marker=".")
    if cfg.include_salt:
        axs[2].scatter(rho_salt, z_salt, c="Gray", alpha=0.1, marker=".")
    axs[2].plot(rpm["shale_rho"], rpm["z"], label="Shale", c="Black", lw=2)
    axs[2].plot(
        rpm["brine_sand_rho"], rpm["z"], label="Brine Sand", c="DodgerBlue", lw=2
    )
    axs[2].plot(rpm["oil_sand_rho"], rpm["z"], label="Oil Sand", c="Seagreen", lw=2)
    axs[2].plot(rpm["gas_sand_rho"], rpm["z"], label="Gas Sand", c="Tomato", lw=2)

    if hasattr(cfg, "bulk_z_shift"):
        axs[0].set_ylim(
            cfg.bulk_z_shift + cfg.digi * cfg.cube_shape[-1], np.min(rpm["z"]) - 100
        )
    else:
        axs[0].set_ylim(
            cfg.digi * cfg.cube_shape[-1], 0
        )  # sharey is True so all subplots use this range
    axs[0].legend()  # put legend in 1st subplot only
    for x in axs:
        x.set_ylabel("Depth bml (m)")
        x.grid(True)
    fig.savefig(os.path.join(cfg.work_subfolder, "QC_RPM_vs_Depth.png"))

    # Crossplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    ai_shale = rho_shale * vp_shale
    si_shale = rho_shale * vs_shale
    vpvs_shale = vp_shale / vs_shale
    pr_shale = (vp_shale**2.0 - 2.0 * vs_shale**2) / (2.0 * (vp_shale**2 - vs_shale**2))
    axs[0].scatter(
        ai_shale[::10],
        si_shale[::10],
        label="Shale",
        c="Saddlebrown",
        alpha=0.1,
        marker=".",
    )
    axs[1].scatter(
        ai_shale[::10],
        vpvs_shale[::10],
        label="Shale",
        c="Saddlebrown",
        alpha=0.1,
        marker=".",
    )
    axs[2].scatter(
        ai_shale[::10],
        pr_shale[::10],
        label="Shale",
        c="Saddlebrown",
        alpha=0.1,
        marker=".",
    )
    axs[0].plot(
        rpm["shale_rho"] * rpm["shale_vp"],
        rpm["shale_rho"] * rpm["shale_vs"],
        label="Shale",
        c="Black",
        lw=2,
    )
    axs[1].plot(
        rpm["shale_rho"] * rpm["shale_vp"],
        rpm["shale_vp"] / rpm["shale_vs"],
        label="Shale",
        c="Black",
        lw=2,
    )
    axs[2].plot(
        rpm["shale_rho"] * rpm["shale_vp"],
        (rpm["shale_vp"] ** 2.0 - 2.0 * rpm["shale_vs"] ** 2)
        / (2.0 * (rpm["shale_vp"] ** 2 - rpm["shale_vs"] ** 2)),
        label="Shale",
        c="Black",
        lw=2,
    )

    ai_sand = rho_sand * vp_sand
    si_sand = rho_sand * vs_sand
    vpvs_sand = vp_sand / vs_sand
    pr_sand = (vp_sand**2.0 - 2.0 * vs_sand**2) / (2.0 * (vp_sand**2 - vs_sand**2))
    axs[0].scatter(
        ai_sand, si_sand, label="Brine Sand", c="Blue", alpha=0.1, marker="."
    )
    axs[1].scatter(
        ai_sand, vpvs_sand, label="Brine Sand", c="Blue", alpha=0.1, marker="."
    )
    axs[2].scatter(
        ai_sand, pr_sand, label="Brine Sand", c="Blue", alpha=0.1, marker="."
    )
    axs[0].plot(
        rpm["brine_sand_rho"] * rpm["brine_sand_vp"],
        rpm["brine_sand_rho"] * rpm["brine_sand_vs"],
        label="Brine Sand",
        c="Dodgerblue",
        lw=2,
    )
    axs[1].plot(
        rpm["brine_sand_rho"] * rpm["brine_sand_vp"],
        rpm["brine_sand_vp"] / rpm["brine_sand_vs"],
        label="Brine Sand",
        c="Dodgerblue",
        lw=2,
    )
    axs[2].plot(
        rpm["brine_sand_rho"] * rpm["brine_sand_vp"],
        (rpm["brine_sand_vp"] ** 2.0 - 2.0 * rpm["brine_sand_vs"] ** 2)
        / (2.0 * (rpm["brine_sand_vp"] ** 2 - rpm["brine_sand_vs"] ** 2)),
        label="Brine Sand",
        c="Dodgerblue",
        lw=2,
    )

    if np.max(oil) > 0:
        ai_oil_sand = rho_oil_sand * vp_oil_sand
        si_oil_sand = rho_oil_sand * vs_oil_sand
        vpvs_oil_sand = vp_oil_sand / vs_oil_sand
        pr_oil_sand = (vp_oil_sand**2.0 - 2.0 * vs_oil_sand**2) / (
            2.0 * (vp_oil_sand**2 - vs_oil_sand**2)
        )
        axs[0].scatter(ai_oil_sand, si_oil_sand, c="Green", alpha=0.1, marker=".")
        axs[1].scatter(ai_oil_sand, vpvs_oil_sand, c="Green", alpha=0.1, marker=".")
        axs[2].scatter(ai_oil_sand, pr_oil_sand, c="Green", alpha=0.1, marker=".")
        axs[0].plot(
            rpm["oil_sand_rho"] * rpm["oil_sand_vp"],
            rpm["oil_sand_rho"] * rpm["oil_sand_vs"],
            label="Oil Sand",
            c="Seagreen",
            lw=2,
        )
        axs[1].plot(
            rpm["oil_sand_rho"] * rpm["oil_sand_vp"],
            rpm["oil_sand_vp"] / rpm["oil_sand_vs"],
            label="Oil Sand",
            c="Seagreen",
            lw=2,
        )
        axs[2].plot(
            rpm["oil_sand_rho"] * rpm["oil_sand_vp"],
            (rpm["oil_sand_vp"] ** 2.0 - 2.0 * rpm["oil_sand_vs"] ** 2)
            / (2.0 * (rpm["oil_sand_vp"] ** 2 - rpm["oil_sand_vs"] ** 2)),
            label="Oil Sand",
            c="Seagreen",
            lw=2,
        )

    if np.max(gas) > 0:
        ai_gas_sand = rho_gas_sand * vp_gas_sand
        si_gas_sand = rho_gas_sand * vs_gas_sand
        vpvs_gas_sand = vp_gas_sand / vs_gas_sand
        pr_gas_sand = (vp_gas_sand**2.0 - 2.0 * vs_gas_sand**2) / (
            2.0 * (vp_gas_sand**2 - vs_gas_sand**2)
        )
        axs[0].scatter(
            ai_gas_sand, si_gas_sand, label="Gas Sand", c="Red", alpha=0.1, marker="."
        )
        axs[1].scatter(
            ai_gas_sand, vpvs_gas_sand, label="Gas Sand", c="Red", alpha=0.1, marker="."
        )
        axs[2].scatter(
            ai_gas_sand, pr_gas_sand, label="Gas Sand", c="Red", alpha=0.1, marker="."
        )
        axs[0].plot(
            rpm["gas_sand_rho"] * rpm["gas_sand_vp"],
            rpm["gas_sand_rho"] * rpm["gas_sand_vs"],
            ":",
            label="Gas Sand",
            c="Tomato",
            lw=2,
        )
        axs[1].plot(
            rpm["gas_sand_rho"] * rpm["gas_sand_vp"],
            rpm["gas_sand_vp"] / rpm["gas_sand_vs"],
            ":",
            label="Gas Sand",
            c="Tomato",
            lw=2,
        )
        axs[2].plot(
            rpm["gas_sand_rho"] * rpm["gas_sand_vp"],
            (rpm["gas_sand_vp"] ** 2.0 - 2.0 * rpm["gas_sand_vs"] ** 2)
            / (2.0 * (rpm["gas_sand_vp"] ** 2 - rpm["gas_sand_vs"] ** 2)),
            label="Gas Sand",
            c="Tomato",
            lw=2,
        )

    axs[0].set_xlabel("AI (vp * rho)")
    axs[0].set_ylabel("SI (vs * rho)")
    axs[1].set_xlabel("AI (vp * rho)")
    axs[1].set_ylabel("Vp/Vs")
    axs[2].set_xlabel("AI (vp * rho)")

    axs[0].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    fig.savefig(os.path.join(cfg.work_subfolder, "QC_RPM_Crossplots.png"))


def clip_vs_via_poissons_ratio(vp, vs):
    """If Vp / Vs < sqrt(2), then clip Vs to sqrt(vp**2 / 2)
    This ensures poissons ratio stays positive
    """
    return np.where(vp / vs < 2**0.5, ((vp**2) / 2) ** 0.5, vs)


def clip_vp_via_poissons_ratio(vp, vs):
    """If Vp / Vs < sqrt(2), then clip Vp to sqrt(2) * vs
    This ensures poissons ratio stays positive
    """
    return np.where(vp / vs < 2**0.5, np.sqrt(2) * vs, vp)


def store_1d_trend_dict_to_hdf(cfg, d, z):
    # Store arrays in HDF
    for k, v in d.items():
        cfg.hdf_init(f"rpm_1d_{k}", shape=z.shape)[:] = v


def calc_ai(rho, vp, log=False):
    ai = rho * vp
    if log:
        ai = 0.5 * np.log(ai)
    return ai


def taper_fn(z, thresh, scalar):
    """Exponential tapering"""
    t = np.ones_like(z, dtype="float")
    t[:thresh] = 1 - np.exp(-scalar * z[:thresh] / thresh)
    return t
