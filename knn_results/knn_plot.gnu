#set terminal postfile       (These commented lines would be used to )
#set output  "d1_plot.ps"    (generate a postscript file.            )
set title "K Nearest Neighbors Results"
set ytics nomirror
set xlabel "Folds"
set ylabel "Accuracy"
plot "knn_validation_lbp_default_training.txt.dat" with lines title "LBP Default" lt rgb "#A40000"
plot "knn_validation_lbp_ror_training.txt.dat" with lines title "LBP ROR" lt rgb "#490206"
plot "knn_validation_lbp_uniform_training.txt.dat" with lines title "LBP Uniform" lt rgb "#008B8B"
plot "knn_validation_lbp_nri_uniform_training.txt.dat" with lines title "LBP NRI_Uniform" lt rgb "#804A00"
plot "knn_validation_glcm_1_training.txt.dat" with lines title "GLCM 1 Distance" lt rgb "#00008B"
plot "knn_validation_glcm_2_training.txt.dat" with lines title "GLCM 2 Distances" lt rgb "#B8860B"
plot "knn_validation_glcm_3_training.txt.dat" with lines title "GLCM 3 Distances" lt rgb "#013220"
plot "knn_validation_glcm_4_training.txt.dat" with lines title "GLCM 4 Distances" lt rgb "#1A2421"
plot "knn_validation_hog_training.txt.dat" with lines title "HOG" lt rgb "#9932CC"
pause -1 "Hit any key to continue"
