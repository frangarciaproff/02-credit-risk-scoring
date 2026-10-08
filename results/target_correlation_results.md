=== near-constant flag columns: TARGET rate by value ===
FLAG_MOBIL
                 n  target_rate
FLAG_MOBIL                     
0                1     0.000000
1           307510     0.080729

FLAG_CONT_MOBILE
                       n  target_rate
FLAG_CONT_MOBILE                     
0                    574     0.078397
1                 306937     0.080733

=== FLAG_DOCUMENT_*: prevalence and TARGET rate when provided ===
                  n_provided  pct_provided  target_rate_when_provided  target_rate_overall
column                                                                                    
FLAG_DOCUMENT_2           13         0.004                   0.307692             0.080729
FLAG_DOCUMENT_3       218340        71.002                   0.088449             0.080729
FLAG_DOCUMENT_4           25         0.008                   0.000000             0.080729
FLAG_DOCUMENT_5         4648         1.511                   0.080034             0.080729
FLAG_DOCUMENT_6        27078         8.806                   0.055654             0.080729
FLAG_DOCUMENT_7           59         0.019                   0.050847             0.080729
FLAG_DOCUMENT_8        25024         8.138                   0.073370             0.080729
FLAG_DOCUMENT_9         1198         0.390                   0.061770             0.080729
FLAG_DOCUMENT_10           7         0.002                   0.000000             0.080729
FLAG_DOCUMENT_11        1203         0.391                   0.062344             0.080729
FLAG_DOCUMENT_12           2         0.001                   0.000000             0.080729
FLAG_DOCUMENT_13        1084         0.353                   0.027675             0.080729
FLAG_DOCUMENT_14         903         0.294                   0.033223             0.080729
FLAG_DOCUMENT_15         372         0.121                   0.029570             0.080729
FLAG_DOCUMENT_16        3053         0.993                   0.049132             0.080729
FLAG_DOCUMENT_17          82         0.027                   0.024390             0.080729
FLAG_DOCUMENT_18        2500         0.813                   0.056800             0.080729
FLAG_DOCUMENT_19         183         0.060                   0.065574             0.080729
FLAG_DOCUMENT_20         156         0.051                   0.083333             0.080729
FLAG_DOCUMENT_21         103         0.033                   0.135922             0.080729

=== DOCUMENTS_PROVIDED_COUNT (sum of FLAG_DOCUMENT_*): TARGET rate by value ===
                               n  target_rate
DOCUMENTS_PROVIDED_COUNT                     
0                          29549      0.05523
1                         270056      0.08449
2                           7742      0.04637
3                            163      0.09816
4                              1      1.00000

=== housing block (_AVG/_MODE/_MEDI) pairwise correlation ===
                         avg_mode  avg_medi  mode_medi
attribute                                             
APARTMENTS               0.973259  0.995081   0.977193
BASEMENTAREA             0.973496  0.994317   0.977938
YEARS_BEGINEXPLUATATION  0.971893  0.993825   0.963539
YEARS_BUILD              0.989444  0.998495   0.989463
COMMONAREA               0.977147  0.995978   0.979887
ELEVATORS                0.978837  0.996099   0.982828
ENTRANCES                0.977743  0.996886   0.980677
FLOORSMAX                0.985689  0.997034   0.988237
FLOORSMIN                0.985875  0.997241   0.988406
LANDAREA                 0.973696  0.991610   0.980835
LIVINGAPARTMENTS         0.970117  0.993825   0.975605
LIVINGAREA               0.972050  0.995596   0.974743
NONLIVINGAPARTMENTS      0.969370  0.990768   0.978575
NONLIVINGAREA            0.966087  0.990444   0.975839

=== AMT_INCOME_TOTAL tail percentiles ===
0.900       270000.0
0.950       337500.0
0.990       472500.0
0.999       900000.0
1.000    117000000.0

=== social-circle columns: tail percentiles ===
OBS_30_CNT_SOCIAL_CIRCLE
0.900      4.0
0.950      6.0
0.990     10.0
0.999     17.0
1.000    348.0

DEF_30_CNT_SOCIAL_CIRCLE
0.900     1.0
0.950     1.0
0.990     2.0
0.999     4.0
1.000    34.0

OBS_60_CNT_SOCIAL_CIRCLE
0.900      4.0
0.950      6.0
0.990     10.0
0.999     16.0
1.000    344.0

DEF_60_CNT_SOCIAL_CIRCLE
0.900     0.0
0.950     1.0
0.990     2.0
0.999     3.0
1.000    24.0