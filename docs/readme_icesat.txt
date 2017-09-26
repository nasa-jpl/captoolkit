readgla.py

- t_sec is secs since 1970-1-1-0-0-0 (originaly, t_sec was secs since 2000,1,1,12,0,0)
- t_year is decimal years
- h_cor is corrected height, and it has been filtered to remove garbage values (e.g. 1.79769313486e+308)
- At read time, provided h_tide and h_load have been removed from h_cor
