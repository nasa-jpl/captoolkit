 ## USING GRID AND SEARCH RADIUS: d 0.5 0.5 r 1
python topofit.py /mnt/devon-r0/shared_data/icesat/floating_/ANT_ICE_ISHELF*.h5 -d 0.5 0.5 -r 1 -i 5 -z 5 -k 1 -m 10 -q 3 -t 2003.5 -j 3031 -v lon lat t_year h_cor -n 2 &
python topofit.py /mnt/devon-r0/shared_data/ers1/floating_/ANT_ER1_ISHELF*.h5 -d 0.5 0.5 -r 1 -i 5 -z 5 -k 1 -m 20 -q 3 -t 1993.5 -j 3031 -v lon lat t_year h_cor -n 2 &
python topofit.py /mnt/devon-r0/shared_data/ers2/floating_/ANT_ER2_ISHELF*.h5 -d 0.5 0.5 -r 1 -i 5 -z 5 -k 1 -m 20 -q 3 -t 1999.0 -j 3031 -v lon lat t_year h_cor -n 2 &
python topofit.py /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2002_2010*.h5 -d 0.5 0.5 -r 1 -i 5 -z 5 -k 1 -m 15 -q 3 -t 2006.0 -j 3031 -v lon lat t_year h_cor -n 2 &
python topofit.py /mnt/devon-r0/shared_data/envisat/floating_/ANT_RA2_ISHELF_2010_2012*.h5 -d 0.5 0.5 -r 1.5 -i 5 -z 5 -k 1 -m 15 -q 3 -t 2011.0 -j 3031 -v lon lat t_year h_cor -n 2 &
python topofit.py /mnt/devon-r0/shared_data/cryosat2/floating_/ANT_CS2_ISHELF*.h5 -d 0.5 0.5 -r 1.5 -i 5 -z 5 -k 1 -m 15 -q 3 -t 2012.0 -j 3031 -v lon lat t_year h_cor -n 2 &
