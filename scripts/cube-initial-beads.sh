density_options="1156.2"
fluid_mass_options="3.0 3.25 3.5"
for density in $density_options; do
  for fluid_mass in $fluid_mass_options; do
    python cube-initial-beads.py --density0 $density --fluid-mass $fluid_mass --kernel-radius 0.00390625
  done
done


density_options="997.28"
fluid_mass_options="3.11"
for density in $density_options; do
  for fluid_mass in $fluid_mass_options; do
    python cube-initial-beads.py --density0 $density --fluid-mass $fluid_mass --kernel-radius 0.00390625
  done
done


density_options="1156.2"
fluid_mass_options="3.11 3.112"
for density in $density_options; do
  for fluid_mass in $fluid_mass_options; do
    python cube-initial-beads.py --density0 $density --fluid-mass $fluid_mass --kernel-radius 0.011
  done
done


density_options="997.28"
fluid_mass_options="3.085"
for density in $density_options; do
  for fluid_mass in $fluid_mass_options; do
    python cube-initial-beads.py --density0 $density --fluid-mass $fluid_mass --kernel-radius 0.011
  done
done

density_options="1156.2"
fluid_mass_options="3.75"
for density in $density_options; do
  for fluid_mass in $fluid_mass_options; do
    python cube-initial-beads.py --density0 $density --fluid-mass $fluid_mass --kernel-radius 0.011
  done
done
