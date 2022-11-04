density_options="1156.2 997.28 1195.48 800 1142 988.05"
fluid_mass_options="2.75 3.0 3.25 3.5 3.75 4"
for density in $density_options; do
  for fluid_mass in $fluid_mass_options; do
    python cube-initial-beads.py --density0 $density --fluid-mass $fluid_mass
  done
done
