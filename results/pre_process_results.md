Shape: 307,511 rows x 122 columns

=== .info() ===
<class 'pandas.DataFrame'>
RangeIndex: 307511 entries, 0 to 307510
Columns: 122 entries, SK_ID_CURR to AMT_REQ_CREDIT_BUREAU_YEAR
dtypes: float64(65), int64(41), str(16)
memory usage: 286.2 MB


=== head ===
   SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  CNT_CHILDREN  AMT_INCOME_TOTAL  AMT_CREDIT  AMT_ANNUITY  AMT_GOODS_PRICE NAME_TYPE_SUITE NAME_INCOME_TYPE            NAME_EDUCATION_TYPE    NAME_FAMILY_STATUS  NAME_HOUSING_TYPE  REGION_POPULATION_RELATIVE  DAYS_BIRTH  DAYS_EMPLOYED  DAYS_REGISTRATION  DAYS_ID_PUBLISH  OWN_CAR_AGE  FLAG_MOBIL  FLAG_EMP_PHONE  FLAG_WORK_PHONE  FLAG_CONT_MOBILE  FLAG_PHONE  FLAG_EMAIL OCCUPATION_TYPE  CNT_FAM_MEMBERS  REGION_RATING_CLIENT  REGION_RATING_CLIENT_W_CITY WEEKDAY_APPR_PROCESS_START  HOUR_APPR_PROCESS_START  REG_REGION_NOT_LIVE_REGION  REG_REGION_NOT_WORK_REGION  LIVE_REGION_NOT_WORK_REGION  REG_CITY_NOT_LIVE_CITY  REG_CITY_NOT_WORK_CITY  LIVE_CITY_NOT_WORK_CITY       ORGANIZATION_TYPE  EXT_SOURCE_1  EXT_SOURCE_2  EXT_SOURCE_3  APARTMENTS_AVG  BASEMENTAREA_AVG  YEARS_BEGINEXPLUATATION_AVG  YEARS_BUILD_AVG  COMMONAREA_AVG  ELEVATORS_AVG  ENTRANCES_AVG  FLOORSMAX_AVG  FLOORSMIN_AVG  LANDAREA_AVG  LIVINGAPARTMENTS_AVG  LIVINGAREA_AVG  NONLIVINGAPARTMENTS_AVG  NONLIVINGAREA_AVG  APARTMENTS_MODE  BASEMENTAREA_MODE  YEARS_BEGINEXPLUATATION_MODE  YEARS_BUILD_MODE  COMMONAREA_MODE  ELEVATORS_MODE  ENTRANCES_MODE  FLOORSMAX_MODE  FLOORSMIN_MODE  LANDAREA_MODE  LIVINGAPARTMENTS_MODE  LIVINGAREA_MODE  NONLIVINGAPARTMENTS_MODE  NONLIVINGAREA_MODE  APARTMENTS_MEDI  BASEMENTAREA_MEDI  YEARS_BEGINEXPLUATATION_MEDI  YEARS_BUILD_MEDI  COMMONAREA_MEDI  ELEVATORS_MEDI  ENTRANCES_MEDI  FLOORSMAX_MEDI  FLOORSMIN_MEDI  LANDAREA_MEDI  LIVINGAPARTMENTS_MEDI  LIVINGAREA_MEDI  NONLIVINGAPARTMENTS_MEDI  NONLIVINGAREA_MEDI FONDKAPREMONT_MODE  HOUSETYPE_MODE  TOTALAREA_MODE WALLSMATERIAL_MODE EMERGENCYSTATE_MODE  OBS_30_CNT_SOCIAL_CIRCLE  DEF_30_CNT_SOCIAL_CIRCLE  OBS_60_CNT_SOCIAL_CIRCLE  DEF_60_CNT_SOCIAL_CIRCLE  DAYS_LAST_PHONE_CHANGE  FLAG_DOCUMENT_2  FLAG_DOCUMENT_3  FLAG_DOCUMENT_4  FLAG_DOCUMENT_5  FLAG_DOCUMENT_6  FLAG_DOCUMENT_7  FLAG_DOCUMENT_8  FLAG_DOCUMENT_9  FLAG_DOCUMENT_10  FLAG_DOCUMENT_11  FLAG_DOCUMENT_12  FLAG_DOCUMENT_13  FLAG_DOCUMENT_14  FLAG_DOCUMENT_15  FLAG_DOCUMENT_16  FLAG_DOCUMENT_17  FLAG_DOCUMENT_18  FLAG_DOCUMENT_19  FLAG_DOCUMENT_20  FLAG_DOCUMENT_21  AMT_REQ_CREDIT_BUREAU_HOUR  AMT_REQ_CREDIT_BUREAU_DAY  AMT_REQ_CREDIT_BUREAU_WEEK  AMT_REQ_CREDIT_BUREAU_MON  AMT_REQ_CREDIT_BUREAU_QRT  AMT_REQ_CREDIT_BUREAU_YEAR
0      100002       1         Cash loans           M            N               Y             0          202500.0    406597.5      24700.5         351000.0   Unaccompanied          Working  Secondary / secondary special  Single / not married  House / apartment                    0.018801       -9461           -637            -3648.0            -2120          NaN           1               1                0                 1           1           0        Laborers              1.0                     2                            2                  WEDNESDAY                       10                           0                           0                            0                       0                       0                        0  Business Entity Type 3      0.083037      0.262949      0.139376          0.0247            0.0369                       0.9722           0.6192          0.0143           0.00         0.0690         0.0833         0.1250        0.0369                0.0202          0.0190                   0.0000             0.0000           0.0252             0.0383                        0.9722            0.6341           0.0144          0.0000          0.0690          0.0833          0.1250         0.0377                  0.022           0.0198                       0.0                 0.0           0.0250             0.0369                        0.9722            0.6243           0.0144            0.00          0.0690          0.0833          0.1250         0.0375                 0.0205           0.0193                    0.0000                0.00   reg oper account  block of flats          0.0149       Stone, brick                  No                       2.0                       2.0                       2.0                       2.0                 -1134.0                0                1                0                0                0                0                0                0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                         0.0                        0.0                         0.0                        0.0                        0.0                         1.0
1      100003       0         Cash loans           F            N               N             0          270000.0   1293502.5      35698.5        1129500.0          Family    State servant               Higher education               Married  House / apartment                    0.003541      -16765          -1188            -1186.0             -291          NaN           1               1                0                 1           1           0      Core staff              2.0                     1                            1                     MONDAY                       11                           0                           0                            0                       0                       0                        0                  School      0.311267      0.622246           NaN          0.0959            0.0529                       0.9851           0.7960          0.0605           0.08         0.0345         0.2917         0.3333        0.0130                0.0773          0.0549                   0.0039             0.0098           0.0924             0.0538                        0.9851            0.8040           0.0497          0.0806          0.0345          0.2917          0.3333         0.0128                  0.079           0.0554                       0.0                 0.0           0.0968             0.0529                        0.9851            0.7987           0.0608            0.08          0.0345          0.2917          0.3333         0.0132                 0.0787           0.0558                    0.0039                0.01   reg oper account  block of flats          0.0714              Block                  No                       1.0                       0.0                       1.0                       0.0                  -828.0                0                1                0                0                0                0                0                0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                         0.0                        0.0                         0.0                        0.0                        0.0                         0.0
2      100004       0    Revolving loans           M            Y               Y             0           67500.0    135000.0       6750.0         135000.0   Unaccompanied          Working  Secondary / secondary special  Single / not married  House / apartment                    0.010032      -19046           -225            -4260.0            -2531         26.0           1               1                1                 1           1           0        Laborers              1.0                     2                            2                     MONDAY                        9                           0                           0                            0                       0                       0                        0              Government           NaN      0.555912      0.729567             NaN               NaN                          NaN              NaN             NaN            NaN            NaN            NaN            NaN           NaN                   NaN             NaN                      NaN                NaN              NaN                NaN                           NaN               NaN              NaN             NaN             NaN             NaN             NaN            NaN                    NaN              NaN                       NaN                 NaN              NaN                NaN                           NaN               NaN              NaN             NaN             NaN             NaN             NaN            NaN                    NaN              NaN                       NaN                 NaN                NaN             NaN             NaN                NaN                 NaN                       0.0                       0.0                       0.0                       0.0                  -815.0                0                0                0                0                0                0                0                0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                         0.0                        0.0                         0.0                        0.0                        0.0                         0.0
3      100006       0         Cash loans           F            N               Y             0          135000.0    312682.5      29686.5         297000.0   Unaccompanied          Working  Secondary / secondary special        Civil marriage  House / apartment                    0.008019      -19005          -3039            -9833.0            -2437          NaN           1               1                0                 1           0           0        Laborers              2.0                     2                            2                  WEDNESDAY                       17                           0                           0                            0                       0                       0                        0  Business Entity Type 3           NaN      0.650442           NaN             NaN               NaN                          NaN              NaN             NaN            NaN            NaN            NaN            NaN           NaN                   NaN             NaN                      NaN                NaN              NaN                NaN                           NaN               NaN              NaN             NaN             NaN             NaN             NaN            NaN                    NaN              NaN                       NaN                 NaN              NaN                NaN                           NaN               NaN              NaN             NaN             NaN             NaN             NaN            NaN                    NaN              NaN                       NaN                 NaN                NaN             NaN             NaN                NaN                 NaN                       2.0                       0.0                       2.0                       0.0                  -617.0                0                1                0                0                0                0                0                0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                         NaN                        NaN                         NaN                        NaN                        NaN                         NaN
4      100007       0         Cash loans           M            N               Y             0          121500.0    513000.0      21865.5         513000.0   Unaccompanied          Working  Secondary / secondary special  Single / not married  House / apartment                    0.028663      -19932          -3038            -4311.0            -3458          NaN           1               1                0                 1           0           0      Core staff              1.0                     2                            2                   THURSDAY                       11                           0                           0                            0                       0                       1                        1                Religion           NaN      0.322738           NaN             NaN               NaN                          NaN              NaN             NaN            NaN            NaN            NaN            NaN           NaN                   NaN             NaN                      NaN                NaN              NaN                NaN                           NaN               NaN              NaN             NaN             NaN             NaN             NaN            NaN                    NaN              NaN                       NaN                 NaN              NaN                NaN                           NaN               NaN              NaN             NaN             NaN             NaN             NaN            NaN                    NaN              NaN                       NaN                 NaN                NaN             NaN             NaN                NaN                 NaN                       0.0                       0.0                       0.0                       0.0                 -1106.0                0                0                0                0                0                0                1                0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                 0                         0.0                        0.0                         0.0                        0.0                        0.0                         0.0

=== dtype split ===
int (41): ['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
float (65): ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_REGISTRATION', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
categorical (16): ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

=== numeric summary (int + float) ===
                                 count           mean            std           min            25%            50%            75%           max
SK_ID_CURR                    307511.0  278180.518577  102790.175348  1.000020e+05  189145.500000  278202.000000  367142.500000  4.562550e+05
TARGET                        307511.0       0.080729       0.272419  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
CNT_CHILDREN                  307511.0       0.417052       0.722121  0.000000e+00       0.000000       0.000000       1.000000  1.900000e+01
DAYS_BIRTH                    307511.0  -16036.995067    4363.988632 -2.522900e+04  -19682.000000  -15750.000000  -12413.000000 -7.489000e+03
DAYS_EMPLOYED                 307511.0   63815.045904  141275.766519 -1.791200e+04   -2760.000000   -1213.000000    -289.000000  3.652430e+05
DAYS_ID_PUBLISH               307511.0   -2994.202373    1509.450419 -7.197000e+03   -4299.000000   -3254.000000   -1720.000000  0.000000e+00
FLAG_MOBIL                    307511.0       0.999997       0.001803  0.000000e+00       1.000000       1.000000       1.000000  1.000000e+00
FLAG_EMP_PHONE                307511.0       0.819889       0.384280  0.000000e+00       1.000000       1.000000       1.000000  1.000000e+00
FLAG_WORK_PHONE               307511.0       0.199368       0.399526  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_CONT_MOBILE              307511.0       0.998133       0.043164  0.000000e+00       1.000000       1.000000       1.000000  1.000000e+00
FLAG_PHONE                    307511.0       0.281066       0.449521  0.000000e+00       0.000000       0.000000       1.000000  1.000000e+00
FLAG_EMAIL                    307511.0       0.056720       0.231307  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
REGION_RATING_CLIENT          307511.0       2.052463       0.509034  1.000000e+00       2.000000       2.000000       2.000000  3.000000e+00
REGION_RATING_CLIENT_W_CITY   307511.0       2.031521       0.502737  1.000000e+00       2.000000       2.000000       2.000000  3.000000e+00
HOUR_APPR_PROCESS_START       307511.0      12.063419       3.265832  0.000000e+00      10.000000      12.000000      14.000000  2.300000e+01
REG_REGION_NOT_LIVE_REGION    307511.0       0.015144       0.122126  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
REG_REGION_NOT_WORK_REGION    307511.0       0.050769       0.219526  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
LIVE_REGION_NOT_WORK_REGION   307511.0       0.040659       0.197499  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
REG_CITY_NOT_LIVE_CITY        307511.0       0.078173       0.268444  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
REG_CITY_NOT_WORK_CITY        307511.0       0.230454       0.421124  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
LIVE_CITY_NOT_WORK_CITY       307511.0       0.179555       0.383817  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_2               307511.0       0.000042       0.006502  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_3               307511.0       0.710023       0.453752  0.000000e+00       0.000000       1.000000       1.000000  1.000000e+00
FLAG_DOCUMENT_4               307511.0       0.000081       0.009016  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_5               307511.0       0.015115       0.122010  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_6               307511.0       0.088055       0.283376  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_7               307511.0       0.000192       0.013850  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_8               307511.0       0.081376       0.273412  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_9               307511.0       0.003896       0.062295  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_10              307511.0       0.000023       0.004771  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_11              307511.0       0.003912       0.062424  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_12              307511.0       0.000007       0.002550  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_13              307511.0       0.003525       0.059268  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_14              307511.0       0.002936       0.054110  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_15              307511.0       0.001210       0.034760  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_16              307511.0       0.009928       0.099144  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_17              307511.0       0.000267       0.016327  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_18              307511.0       0.008130       0.089798  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_19              307511.0       0.000595       0.024387  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_20              307511.0       0.000507       0.022518  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
FLAG_DOCUMENT_21              307511.0       0.000335       0.018299  0.000000e+00       0.000000       0.000000       0.000000  1.000000e+00
AMT_INCOME_TOTAL              307511.0  168797.919297  237123.146279  2.565000e+04  112500.000000  147150.000000  202500.000000  1.170000e+08
AMT_CREDIT                    307511.0  599025.999706  402490.776996  4.500000e+04  270000.000000  513531.000000  808650.000000  4.050000e+06
AMT_ANNUITY                   307499.0   27108.573909   14493.737315  1.615500e+03   16524.000000   24903.000000   34596.000000  2.580255e+05
AMT_GOODS_PRICE               307233.0  538396.207429  369446.460540  4.050000e+04  238500.000000  450000.000000  679500.000000  4.050000e+06
REGION_POPULATION_RELATIVE    307511.0       0.020868       0.013831  2.900000e-04       0.010006       0.018850       0.028663  7.250800e-02
DAYS_REGISTRATION             307511.0   -4986.120328    3522.886321 -2.467200e+04   -7479.500000   -4504.000000   -2010.000000  0.000000e+00
OWN_CAR_AGE                   104582.0      12.061091      11.944812  0.000000e+00       5.000000       9.000000      15.000000  9.100000e+01
CNT_FAM_MEMBERS               307509.0       2.152665       0.910682  1.000000e+00       2.000000       2.000000       3.000000  2.000000e+01
EXT_SOURCE_1                  134133.0       0.502130       0.211062  1.456813e-02       0.334007       0.505998       0.675053  9.626928e-01
EXT_SOURCE_2                  306851.0       0.514393       0.191060  8.173617e-08       0.392457       0.565961       0.663617  8.549997e-01
EXT_SOURCE_3                  246546.0       0.510853       0.194844  5.272652e-04       0.370650       0.535276       0.669057  8.960095e-01
APARTMENTS_AVG                151450.0       0.117440       0.108240  0.000000e+00       0.057700       0.087600       0.148500  1.000000e+00
BASEMENTAREA_AVG              127568.0       0.088442       0.082438  0.000000e+00       0.044200       0.076300       0.112200  1.000000e+00
YEARS_BEGINEXPLUATATION_AVG   157504.0       0.977735       0.059223  0.000000e+00       0.976700       0.981600       0.986600  1.000000e+00
YEARS_BUILD_AVG               103023.0       0.752471       0.113280  0.000000e+00       0.687200       0.755200       0.823200  1.000000e+00
COMMONAREA_AVG                 92646.0       0.044621       0.076036  0.000000e+00       0.007800       0.021100       0.051500  1.000000e+00
ELEVATORS_AVG                 143620.0       0.078942       0.134576  0.000000e+00       0.000000       0.000000       0.120000  1.000000e+00
ENTRANCES_AVG                 152683.0       0.149725       0.100049  0.000000e+00       0.069000       0.137900       0.206900  1.000000e+00
FLOORSMAX_AVG                 154491.0       0.226282       0.144641  0.000000e+00       0.166700       0.166700       0.333300  1.000000e+00
FLOORSMIN_AVG                  98869.0       0.231894       0.161380  0.000000e+00       0.083300       0.208300       0.375000  1.000000e+00
LANDAREA_AVG                  124921.0       0.066333       0.081184  0.000000e+00       0.018700       0.048100       0.085600  1.000000e+00
LIVINGAPARTMENTS_AVG           97312.0       0.100775       0.092576  0.000000e+00       0.050400       0.075600       0.121000  1.000000e+00
LIVINGAREA_AVG                153161.0       0.107399       0.110565  0.000000e+00       0.045300       0.074500       0.129900  1.000000e+00
NONLIVINGAPARTMENTS_AVG        93997.0       0.008809       0.047732  0.000000e+00       0.000000       0.000000       0.003900  1.000000e+00
NONLIVINGAREA_AVG             137829.0       0.028358       0.069523  0.000000e+00       0.000000       0.003600       0.027700  1.000000e+00
APARTMENTS_MODE               151450.0       0.114231       0.107936  0.000000e+00       0.052500       0.084000       0.143900  1.000000e+00
BASEMENTAREA_MODE             127568.0       0.087543       0.084307  0.000000e+00       0.040700       0.074600       0.112400  1.000000e+00
YEARS_BEGINEXPLUATATION_MODE  157504.0       0.977065       0.064575  0.000000e+00       0.976700       0.981600       0.986600  1.000000e+00
YEARS_BUILD_MODE              103023.0       0.759637       0.110111  0.000000e+00       0.699400       0.764800       0.823600  1.000000e+00
COMMONAREA_MODE                92646.0       0.042553       0.074445  0.000000e+00       0.007200       0.019000       0.049000  1.000000e+00
ELEVATORS_MODE                143620.0       0.074490       0.132256  0.000000e+00       0.000000       0.000000       0.120800  1.000000e+00
ENTRANCES_MODE                152683.0       0.145193       0.100977  0.000000e+00       0.069000       0.137900       0.206900  1.000000e+00
FLOORSMAX_MODE                154491.0       0.222315       0.143709  0.000000e+00       0.166700       0.166700       0.333300  1.000000e+00
FLOORSMIN_MODE                 98869.0       0.228058       0.161160  0.000000e+00       0.083300       0.208300       0.375000  1.000000e+00
LANDAREA_MODE                 124921.0       0.064958       0.081750  0.000000e+00       0.016600       0.045800       0.084100  1.000000e+00
LIVINGAPARTMENTS_MODE          97312.0       0.105645       0.097880  0.000000e+00       0.054200       0.077100       0.131300  1.000000e+00
LIVINGAREA_MODE               153161.0       0.105975       0.111845  0.000000e+00       0.042700       0.073100       0.125200  1.000000e+00
NONLIVINGAPARTMENTS_MODE       93997.0       0.008076       0.046276  0.000000e+00       0.000000       0.000000       0.003900  1.000000e+00
NONLIVINGAREA_MODE            137829.0       0.027022       0.070254  0.000000e+00       0.000000       0.001100       0.023100  1.000000e+00
APARTMENTS_MEDI               151450.0       0.117850       0.109076  0.000000e+00       0.058300       0.086400       0.148900  1.000000e+00
BASEMENTAREA_MEDI             127568.0       0.087955       0.082179  0.000000e+00       0.043700       0.075800       0.111600  1.000000e+00
YEARS_BEGINEXPLUATATION_MEDI  157504.0       0.977752       0.059897  0.000000e+00       0.976700       0.981600       0.986600  1.000000e+00
YEARS_BUILD_MEDI              103023.0       0.755746       0.112066  0.000000e+00       0.691400       0.758500       0.825600  1.000000e+00
COMMONAREA_MEDI                92646.0       0.044595       0.076144  0.000000e+00       0.007900       0.020800       0.051300  1.000000e+00
ELEVATORS_MEDI                143620.0       0.078078       0.134467  0.000000e+00       0.000000       0.000000       0.120000  1.000000e+00
ENTRANCES_MEDI                152683.0       0.149213       0.100368  0.000000e+00       0.069000       0.137900       0.206900  1.000000e+00
FLOORSMAX_MEDI                154491.0       0.225897       0.145067  0.000000e+00       0.166700       0.166700       0.333300  1.000000e+00
FLOORSMIN_MEDI                 98869.0       0.231625       0.161934  0.000000e+00       0.083300       0.208300       0.375000  1.000000e+00
LANDAREA_MEDI                 124921.0       0.067169       0.082167  0.000000e+00       0.018700       0.048700       0.086800  1.000000e+00
LIVINGAPARTMENTS_MEDI          97312.0       0.101954       0.093642  0.000000e+00       0.051300       0.076100       0.123100  1.000000e+00
LIVINGAREA_MEDI               153161.0       0.108607       0.112260  0.000000e+00       0.045700       0.074900       0.130300  1.000000e+00
NONLIVINGAPARTMENTS_MEDI       93997.0       0.008651       0.047415  0.000000e+00       0.000000       0.000000       0.003900  1.000000e+00
NONLIVINGAREA_MEDI            137829.0       0.028236       0.070166  0.000000e+00       0.000000       0.003100       0.026600  1.000000e+00
TOTALAREA_MODE                159080.0       0.102547       0.107462  0.000000e+00       0.041200       0.068800       0.127600  1.000000e+00
OBS_30_CNT_SOCIAL_CIRCLE      306490.0       1.422245       2.400989  0.000000e+00       0.000000       0.000000       2.000000  3.480000e+02
DEF_30_CNT_SOCIAL_CIRCLE      306490.0       0.143421       0.446698  0.000000e+00       0.000000       0.000000       0.000000  3.400000e+01
OBS_60_CNT_SOCIAL_CIRCLE      306490.0       1.405292       2.379803  0.000000e+00       0.000000       0.000000       2.000000  3.440000e+02
DEF_60_CNT_SOCIAL_CIRCLE      306490.0       0.100049       0.362291  0.000000e+00       0.000000       0.000000       0.000000  2.400000e+01
DAYS_LAST_PHONE_CHANGE        307510.0    -962.858788     826.808487 -4.292000e+03   -1570.000000    -757.000000    -274.000000  0.000000e+00
AMT_REQ_CREDIT_BUREAU_HOUR    265992.0       0.006402       0.083849  0.000000e+00       0.000000       0.000000       0.000000  4.000000e+00
AMT_REQ_CREDIT_BUREAU_DAY     265992.0       0.007000       0.110757  0.000000e+00       0.000000       0.000000       0.000000  9.000000e+00
AMT_REQ_CREDIT_BUREAU_WEEK    265992.0       0.034362       0.204685  0.000000e+00       0.000000       0.000000       0.000000  8.000000e+00
AMT_REQ_CREDIT_BUREAU_MON     265992.0       0.267395       0.916002  0.000000e+00       0.000000       0.000000       0.000000  2.700000e+01
AMT_REQ_CREDIT_BUREAU_QRT     265992.0       0.265474       0.794056  0.000000e+00       0.000000       0.000000       0.000000  2.610000e+02
AMT_REQ_CREDIT_BUREAU_YEAR    265992.0       1.899974       1.869295  0.000000e+00       0.000000       1.000000       3.000000  2.500000e+01

=== categorical summary ===
                             count unique                            top    freq
NAME_CONTRACT_TYPE          307511      2                     Cash loans  278232
CODE_GENDER                 307511      3                              F  202448
FLAG_OWN_CAR                307511      2                              N  202924
FLAG_OWN_REALTY             307511      2                              Y  213312
NAME_TYPE_SUITE             306219      7                  Unaccompanied  248526
NAME_INCOME_TYPE            307511      8                        Working  158774
NAME_EDUCATION_TYPE         307511      5  Secondary / secondary special  218391
NAME_FAMILY_STATUS          307511      6                        Married  196432
NAME_HOUSING_TYPE           307511      6              House / apartment  272868
OCCUPATION_TYPE             211120     18                       Laborers   55186
WEEKDAY_APPR_PROCESS_START  307511      7                        TUESDAY   53901
ORGANIZATION_TYPE           307511     58         Business Entity Type 3   67992
FONDKAPREMONT_MODE           97216      4               reg oper account   73830
HOUSETYPE_MODE              153214      3                 block of flats  150503
WALLSMATERIAL_MODE          151170      7                          Panel   66040
EMERGENCYSTATE_MODE         161756      2                             No  159428

=== cardinality (nunique per column, ascending) ===
                              nunique
TARGET                              2
NAME_CONTRACT_TYPE                  2
FLAG_OWN_REALTY                     2
FLAG_OWN_CAR                        2
FLAG_WORK_PHONE                     2
FLAG_CONT_MOBILE                    2
FLAG_EMP_PHONE                      2
FLAG_MOBIL                          2
FLAG_PHONE                          2
FLAG_EMAIL                          2
LIVE_CITY_NOT_WORK_CITY             2
REG_CITY_NOT_WORK_CITY              2
LIVE_REGION_NOT_WORK_REGION         2
REG_CITY_NOT_LIVE_CITY              2
REG_REGION_NOT_LIVE_REGION          2
REG_REGION_NOT_WORK_REGION          2
FLAG_DOCUMENT_18                    2
FLAG_DOCUMENT_19                    2
FLAG_DOCUMENT_20                    2
FLAG_DOCUMENT_21                    2
FLAG_DOCUMENT_6                     2
FLAG_DOCUMENT_7                     2
FLAG_DOCUMENT_8                     2
FLAG_DOCUMENT_9                     2
FLAG_DOCUMENT_10                    2
FLAG_DOCUMENT_11                    2
FLAG_DOCUMENT_12                    2
FLAG_DOCUMENT_13                    2
FLAG_DOCUMENT_14                    2
FLAG_DOCUMENT_15                    2
FLAG_DOCUMENT_16                    2
EMERGENCYSTATE_MODE                 2
FLAG_DOCUMENT_2                     2
FLAG_DOCUMENT_3                     2
FLAG_DOCUMENT_4                     2
FLAG_DOCUMENT_5                     2
FLAG_DOCUMENT_17                    2
CODE_GENDER                         3
REGION_RATING_CLIENT_W_CITY         3
REGION_RATING_CLIENT                3
HOUSETYPE_MODE                      3
FONDKAPREMONT_MODE                  4
AMT_REQ_CREDIT_BUREAU_HOUR          5
NAME_EDUCATION_TYPE                 5
NAME_FAMILY_STATUS                  6
NAME_HOUSING_TYPE                   6
NAME_TYPE_SUITE                     7
WALLSMATERIAL_MODE                  7
WEEKDAY_APPR_PROCESS_START          7
NAME_INCOME_TYPE                    8
DEF_60_CNT_SOCIAL_CIRCLE            9
AMT_REQ_CREDIT_BUREAU_DAY           9
AMT_REQ_CREDIT_BUREAU_WEEK          9
DEF_30_CNT_SOCIAL_CIRCLE           10
AMT_REQ_CREDIT_BUREAU_QRT          11
CNT_CHILDREN                       15
CNT_FAM_MEMBERS                    17
OCCUPATION_TYPE                    18
HOUR_APPR_PROCESS_START            24
AMT_REQ_CREDIT_BUREAU_MON          24
FLOORSMAX_MODE                     25
FLOORSMIN_MODE                     25
AMT_REQ_CREDIT_BUREAU_YEAR         25
ELEVATORS_MODE                     26
ENTRANCES_MODE                     30
OBS_30_CNT_SOCIAL_CIRCLE           33
OBS_60_CNT_SOCIAL_CIRCLE           33
ENTRANCES_MEDI                     46
ELEVATORS_MEDI                     46
FLOORSMIN_MEDI                     47
FLOORSMAX_MEDI                     49
ORGANIZATION_TYPE                  58
OWN_CAR_AGE                        62
REGION_POPULATION_RELATIVE         81
YEARS_BUILD_AVG                   149
YEARS_BUILD_MEDI                  151
YEARS_BUILD_MODE                  154
NONLIVINGAPARTMENTS_MODE          167
NONLIVINGAPARTMENTS_MEDI          214
YEARS_BEGINEXPLUATATION_MODE      221
YEARS_BEGINEXPLUATATION_MEDI      245
ELEVATORS_AVG                     257
YEARS_BEGINEXPLUATATION_AVG       285
ENTRANCES_AVG                     285
FLOORSMIN_AVG                     305
NONLIVINGAPARTMENTS_AVG           386
FLOORSMAX_AVG                     403
LIVINGAPARTMENTS_MODE             736
APARTMENTS_MODE                   760
EXT_SOURCE_3                      814
AMT_GOODS_PRICE                  1002
LIVINGAPARTMENTS_MEDI            1097
APARTMENTS_MEDI                  1148
LIVINGAPARTMENTS_AVG             1868
APARTMENTS_AVG                   2339
AMT_INCOME_TOTAL                 2548
COMMONAREA_MODE                  3128
COMMONAREA_AVG                   3181
COMMONAREA_MEDI                  3202
NONLIVINGAREA_AVG                3290
NONLIVINGAREA_MEDI               3323
NONLIVINGAREA_MODE               3327
LANDAREA_AVG                     3527
LANDAREA_MEDI                    3560
LANDAREA_MODE                    3563
BASEMENTAREA_MEDI                3772
DAYS_LAST_PHONE_CHANGE           3773
BASEMENTAREA_AVG                 3780
BASEMENTAREA_MODE                3841
TOTALAREA_MODE                   5116
LIVINGAREA_AVG                   5199
LIVINGAREA_MEDI                  5281
LIVINGAREA_MODE                  5301
AMT_CREDIT                       5603
DAYS_ID_PUBLISH                  6168
DAYS_EMPLOYED                   12574
AMT_ANNUITY                     13672
DAYS_REGISTRATION               15688
DAYS_BIRTH                      17460
EXT_SOURCE_1                   114584
EXT_SOURCE_2                   119831
SK_ID_CURR                     307511

=== missing values (67 of 122 columns) ===
                              n_missing  pct_missing
COMMONAREA_MEDI                  214865        69.87
COMMONAREA_MODE                  214865        69.87
COMMONAREA_AVG                   214865        69.87
NONLIVINGAPARTMENTS_MODE         213514        69.43
NONLIVINGAPARTMENTS_MEDI         213514        69.43
NONLIVINGAPARTMENTS_AVG          213514        69.43
FONDKAPREMONT_MODE               210295        68.39
LIVINGAPARTMENTS_AVG             210199        68.35
LIVINGAPARTMENTS_MEDI            210199        68.35
LIVINGAPARTMENTS_MODE            210199        68.35
FLOORSMIN_MEDI                   208642        67.85
FLOORSMIN_MODE                   208642        67.85
FLOORSMIN_AVG                    208642        67.85
YEARS_BUILD_MODE                 204488        66.50
YEARS_BUILD_MEDI                 204488        66.50
YEARS_BUILD_AVG                  204488        66.50
OWN_CAR_AGE                      202929        65.99
LANDAREA_AVG                     182590        59.38
LANDAREA_MEDI                    182590        59.38
LANDAREA_MODE                    182590        59.38
BASEMENTAREA_MODE                179943        58.52
BASEMENTAREA_MEDI                179943        58.52
BASEMENTAREA_AVG                 179943        58.52
EXT_SOURCE_1                     173378        56.38
NONLIVINGAREA_MEDI               169682        55.18
NONLIVINGAREA_AVG                169682        55.18
NONLIVINGAREA_MODE               169682        55.18
ELEVATORS_MEDI                   163891        53.30
ELEVATORS_MODE                   163891        53.30
ELEVATORS_AVG                    163891        53.30
WALLSMATERIAL_MODE               156341        50.84
APARTMENTS_AVG                   156061        50.75
APARTMENTS_MEDI                  156061        50.75
APARTMENTS_MODE                  156061        50.75
ENTRANCES_MODE                   154828        50.35
ENTRANCES_MEDI                   154828        50.35
ENTRANCES_AVG                    154828        50.35
LIVINGAREA_AVG                   154350        50.19
LIVINGAREA_MODE                  154350        50.19
LIVINGAREA_MEDI                  154350        50.19
HOUSETYPE_MODE                   154297        50.18
FLOORSMAX_AVG                    153020        49.76
FLOORSMAX_MEDI                   153020        49.76
FLOORSMAX_MODE                   153020        49.76
YEARS_BEGINEXPLUATATION_MODE     150007        48.78
YEARS_BEGINEXPLUATATION_MEDI     150007        48.78
YEARS_BEGINEXPLUATATION_AVG      150007        48.78
TOTALAREA_MODE                   148431        48.27
EMERGENCYSTATE_MODE              145755        47.40
OCCUPATION_TYPE                   96391        31.35
EXT_SOURCE_3                      60965        19.83
AMT_REQ_CREDIT_BUREAU_WEEK        41519        13.50
AMT_REQ_CREDIT_BUREAU_MON         41519        13.50
AMT_REQ_CREDIT_BUREAU_HOUR        41519        13.50
AMT_REQ_CREDIT_BUREAU_YEAR        41519        13.50
AMT_REQ_CREDIT_BUREAU_QRT         41519        13.50
AMT_REQ_CREDIT_BUREAU_DAY         41519        13.50
NAME_TYPE_SUITE                    1292         0.42
DEF_30_CNT_SOCIAL_CIRCLE           1021         0.33
DEF_60_CNT_SOCIAL_CIRCLE           1021         0.33
OBS_30_CNT_SOCIAL_CIRCLE           1021         0.33
OBS_60_CNT_SOCIAL_CIRCLE           1021         0.33
EXT_SOURCE_2                        660         0.21
AMT_GOODS_PRICE                     278         0.09
AMT_ANNUITY                          12         0.00
CNT_FAM_MEMBERS                       2         0.00
DAYS_LAST_PHONE_CHANGE                1         0.00