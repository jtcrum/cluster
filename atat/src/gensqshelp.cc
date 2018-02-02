const char *helpstring=""
"IMPORTANT NOTE: Please see mcsqs command for a better, easier-to-use SQS generator.\n"
"\n"
"This code requires 3 input files:\n"
"1) A lattice file (by default lat.in) in the same format as for\n"
"   maps or corrdump.\n"
"2) A cluster file (by default clusters.out), as generated with\n"
"    the corrdump utility.\n"
"3) A target correlation file (by default tcorr.out) which\n"
"   contains the value of desired correlations for each of\n"
"   the clusters listed in the cluster file.\n"
"\n"
"A typical caling sequence would be:\n"
"\n"
"# the following command can be used to generate the target correlation file tcorr.out\n"
"\n"
"corrdump -noe -2=maxradius -rnd -s=conc.in > tcorr.out\n"
"\n"
"# where maxradius is the length of the longest pair desired\n"
"# and where conc.in contains an (ordered) structure having the\n"
"# desired concentration.\n"
"# The geometry of the structure in the conc.in file is not crucial -\n"
"# only the average concentration on each sublattice will be used.\n"
"# CAUTION: Here, a 'sublattice' is a set of symmetrically equivalent point clusters.\n"
"# If your system contains multiple sublattices (as evidenced by multiple\n"
"# point clusters in clusters.out, make sure that your conc.in file sets\n"
"# the composition of each sublattice correctly! This can be verified\n"
"# by looking at the point correlations output.\n"
"\n"
"#this looks for possible sqs of 8 atoms/ cell\n"
"gensqs -n=8 > sqs8.out\n"
"\n"
"corrdump -2=anotherradius -3=anotherradius  -noe -s=sqs8.out\n"
"# this helps you decide which sqs is best based on other correlations\n"
"# associated with clusters (pairs and triplets) of diamter less than\n"
"# anotherradius.\n"
"\n"
"Caution:\n"
" gensqs only generates structures containing exactly the number of atoms per unit cell specified by the -n option.\n"
" (If an SQS with a smaller unit cell exists, it will not be listed.)\n"
" If you give too many correlations to match, the code may not\n"
" output anything.\n"
" Finding an 8-atom sqs takes a few minutes, an 16-atom sqs, a few hours\n"
" and a 32-atom sqs, a few days!\n"
" The exact speed depends on the symmetry of the lattice and on your\n"
" computer.\n"
;