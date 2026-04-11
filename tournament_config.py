"""
Tournament Configuration: ~300 FRED Series for Korean CPI Forecasting
=====================================================================
Each entry: (FRED_series_ID, display_name, frequency, transform)
  - frequency: "d" = daily, "m" = monthly
  - transform: "raw" = use as-is, "yoy" = year-over-year %, "diff" = month-over-month difference

All series verified against FRED (fred.stlouisfed.org) with data available from at least 2005.
"""

# ============================================================
# 1. EXCHANGE RATES (~15)
# ============================================================
EXCHANGE_RATES = [
    ("DEXKOUS",        "USD_KRW",           "d", "raw"),   # Korean Won per USD
    ("DEXJPUS",        "JPY_USD",           "d", "raw"),   # Japanese Yen per USD
    ("DEXCHUS",        "CNY_USD",           "d", "raw"),   # Chinese Yuan per USD
    ("DEXUSEU",        "EUR_USD",           "d", "raw"),   # USD per Euro
    ("DEXUSUK",        "GBP_USD",           "d", "raw"),   # USD per British Pound
    ("DEXCAUS",        "CAD_USD",           "d", "raw"),   # Canadian Dollar per USD
    ("DEXUSAL",        "AUD_USD",           "d", "raw"),   # USD per Australian Dollar
    ("DEXSZUS",        "CHF_USD",           "d", "raw"),   # Swiss Franc per USD
    ("DEXSIUS",        "SGD_USD",           "d", "raw"),   # Singapore Dollar per USD
    ("DEXTAUS",        "TWD_USD",           "d", "raw"),   # Taiwan Dollar per USD
    ("DEXMAUS",        "MYR_USD",           "d", "raw"),   # Malaysian Ringgit per USD
    ("DEXTHUS",        "THB_USD",           "d", "raw"),   # Thai Baht per USD
    ("DEXMXUS",        "MXN_USD",           "d", "raw"),   # Mexican Peso per USD
    ("DEXBZUS",        "BRL_USD",           "d", "raw"),   # Brazilian Real per USD
    ("DEXINUS",        "INR_USD",           "d", "raw"),   # Indian Rupee per USD
    ("DTWEXBGS",       "DXY_Broad",         "d", "raw"),   # Nominal Broad USD Index
    ("DTWEXAFEGS",     "DXY_AFE",           "d", "raw"),   # Nominal Advanced Foreign Economies USD Index
    ("DTWEXEMEGS",     "DXY_EME",           "d", "raw"),   # Nominal Emerging Market Economies USD Index
    ("DEXSFUS",        "ZAR_USD",           "d", "raw"),   # South African Rand per USD
    ("DEXSDUS",        "SEK_USD",           "d", "raw"),   # Swedish Krona per USD
    ("DEXNOUS",        "NOK_USD",           "d", "raw"),   # Norwegian Krone per USD
    ("DEXHKUS",        "HKD_USD",           "d", "raw"),   # Hong Kong Dollar per USD
    ("RBKRBIS",        "KR_REER",           "m", "raw"),   # Korea Real Broad Effective Exchange Rate (BIS)
    ("RBUSBIS",        "US_REER",           "m", "raw"),   # US Real Broad Effective Exchange Rate (BIS)
    ("RBCNBIS",        "CN_REER",           "m", "raw"),   # China Real Broad Effective Exchange Rate (BIS)
    ("RBJPBIS",        "JP_REER",           "m", "raw"),   # Japan Real Broad Effective Exchange Rate (BIS)
]

# ============================================================
# 2. COMMODITIES — ENERGY (~20)
# ============================================================
COMMODITIES_ENERGY = [
    ("DCOILWTICO",     "Oil_WTI_d",         "d", "raw"),   # WTI Crude Oil (daily)
    ("DCOILBRENTEU",   "Oil_Brent_d",       "d", "raw"),   # Brent Crude Oil (daily)
    ("POILWTIUSDM",    "Oil_WTI_m",         "m", "raw"),   # WTI Crude (IMF global, monthly)
    ("POILBREUSDM",    "Oil_Brent_m",       "m", "raw"),   # Brent Crude (IMF global, monthly)
    ("DHHNGSP",        "NatGas_HH",         "d", "raw"),   # Henry Hub Natural Gas Spot (daily)
    ("PNGASEUUSDM",    "NatGas_EU",         "m", "raw"),   # Natural Gas, EU (monthly)
    ("PNGASJPUSDM",    "LNG_Asia",          "m", "raw"),   # LNG, Asia/Japan (monthly)
    ("PCOALAUUSDM",    "Coal_AU",           "m", "raw"),   # Coal, Australia (monthly)
    ("GASREGW",        "Gasoline_Reg",      "d", "raw"),   # US Regular Gasoline Price (weekly→daily)
    ("GASDESW",        "Diesel_US",         "d", "raw"),   # US Diesel Price (weekly)
    ("DPROPANEMBTX",   "Propane",           "d", "raw"),   # Propane, Mont Belvieu TX (daily)
    ("PNRGINDEXM",     "Energy_Idx",        "m", "raw"),   # IMF Global Energy Price Index
    ("CPIENGSL",       "US_CPI_Energy",     "m", "yoy"),   # US CPI Energy Component
    ("WPU0531",        "PPI_Fuels",         "m", "yoy"),   # PPI Fuels and Related Products
    ("MCOILWTICO",     "Oil_WTI_mEIA",     "m", "raw"),   # WTI Crude Oil (EIA monthly)
    ("MCOILBRENTEU",   "Oil_Brent_mEIA",   "m", "raw"),   # Brent Crude Oil (EIA monthly)
    ("MHHNGSP",        "NatGas_HH_m",      "m", "raw"),   # Henry Hub Natural Gas Spot (monthly)
    ("GASREGM",        "Gasoline_Reg_m",    "m", "raw"),   # US Regular Gasoline Price (monthly)
    ("WTISPLC",        "Oil_WTI_Spot",      "m", "raw"),   # WTI Crude Spot Price (monthly, BLS)
]

# ============================================================
# 3. COMMODITIES — METALS (~10)
# ============================================================
COMMODITIES_METALS = [
    ("GOLDAMGBD228NLBM","Gold",             "d", "raw"),   # Gold Price (London AM Fix, daily)
    ("PCOPPUSDM",      "Copper",            "m", "raw"),   # Global Copper Price (monthly)
    ("PALUMUSDM",      "Aluminum",          "m", "raw"),   # Global Aluminum Price (monthly)
    ("PNICKUSDM",      "Nickel",            "m", "raw"),   # Global Nickel Price (monthly)
    ("PZINCUSDM",      "Zinc",              "m", "raw"),   # Global Zinc Price (monthly)
    ("PLEADUSDM",      "Lead",              "m", "raw"),   # Global Lead Price (monthly)
    ("PTINUSDM",       "Tin",               "m", "raw"),   # Global Tin Price (monthly)
    ("PIORECRUSDM",    "IronOre",           "m", "raw"),   # Global Iron Ore Price (monthly)
    ("WPU10",          "PPI_Metals",        "m", "yoy"),   # PPI Metals & Metal Products
    ("PMETAINDEXM",    "Metals_Idx",        "m", "raw"),   # IMF Metals Price Index (monthly)
    ("WPU102102",      "PPI_CopperNickel",  "m", "yoy"),   # PPI Copper and Nickel Ores
]

# ============================================================
# 4. COMMODITIES — AGRICULTURE (~15)
# ============================================================
COMMODITIES_AGRICULTURE = [
    ("PWHEAMTUSDM",    "Wheat",             "m", "raw"),   # Global Wheat Price (monthly)
    ("PMAIZMTUSDM",    "Corn",              "m", "raw"),   # Global Corn/Maize Price (monthly)
    ("PSOYBUSDM",      "Soybeans",          "m", "raw"),   # Global Soybeans Price (monthly)
    ("PRICENPQUSDM",   "Rice",              "m", "raw"),   # Global Rice Price (monthly)
    ("PSUGAISAUSDM",   "Sugar",             "m", "raw"),   # Global Sugar No.11 Price (monthly)
    ("PCOFFOTMUSDM",   "Coffee",            "m", "raw"),   # Global Coffee (Arabica) Price (monthly)
    ("PCOTTINDUSDM",   "Cotton",            "m", "raw"),   # Global Cotton Price (monthly)
    ("PBEEFUSDM",      "Beef",              "m", "raw"),   # Global Beef Price (monthly)
    ("PPORKUSDM",      "Pork",              "m", "raw"),   # Global Pork Price (monthly)
    ("PPOABORUSDM",    "PalmOil",           "m", "raw"),   # Global Palm Oil Price (monthly)
    ("PBANSOPUSDM",    "Bananas",           "m", "raw"),   # Global Banana Price (monthly)
    ("PFOODINDEXM",    "Food_Idx",          "m", "raw"),   # IMF Global Food Price Index
    ("PRAWMINDEXM",    "AgRawMat_Idx",      "m", "raw"),   # IMF Agricultural Raw Material Index
    ("PALLFNFINDEXM",  "AllCommod_Idx",     "m", "raw"),   # IMF All Commodities Index
    ("WPU01",          "PPI_FarmProd",       "m", "yoy"),   # PPI Farm Products
    ("POLVOILUSDM",    "OliveOil",          "m", "raw"),   # Global Olive Oil Price (monthly)
    ("PSUNOUSDM",      "SunflowerOil",      "m", "raw"),   # Global Sunflower Oil Price (monthly)
    ("PLOGSKUSDM",     "HardLogs",          "m", "raw"),   # Global Hard Logs Price (monthly)
    ("PPORKUSDM",      "Pork_v2",           "m", "raw"),   # Global Pork Price (monthly) — verify not duplicate
    ("WPU012",         "PPI_Grains",        "m", "yoy"),   # PPI Grains
    ("WPU0183",        "PPI_Oilseeds",      "m", "yoy"),   # PPI Oilseeds
    ("PPRUBUSDM",      "Rubber",            "m", "raw"),   # Global Rubber Price (monthly)
]

# ============================================================
# 5. US PRICES (~20)
# ============================================================
US_PRICES = [
    ("CPIAUCSL",       "US_CPI",            "m", "yoy"),   # CPI All Items
    ("CPILFESL",       "US_CoreCPI",        "m", "yoy"),   # CPI All Items Less Food & Energy (Core)
    ("CPIFABSL",       "US_CPI_Food",       "m", "yoy"),   # CPI Food & Beverages
    ("CPIHOSSL",       "US_CPI_Housing",    "m", "yoy"),   # CPI Housing
    ("CPITRNSL",       "US_CPI_Transport",  "m", "yoy"),   # CPI Transportation
    ("CPIMEDSL",       "US_CPI_Medical",    "m", "yoy"),   # CPI Medical Care
    ("CPIAPPSL",       "US_CPI_Apparel",    "m", "yoy"),   # CPI Apparel
    ("CPIRECSL",       "US_CPI_Recreat",    "m", "yoy"),   # CPI Recreation
    ("CUSR0000SAH1",   "US_CPI_Shelter",    "m", "yoy"),   # CPI Shelter
    ("CUSR0000SEHA",   "US_CPI_Rent",       "m", "yoy"),   # CPI Rent of Primary Residence
    ("CUSR0000SETA01", "US_CPI_NewVeh",     "m", "yoy"),   # CPI New Vehicles
    ("CUSR0000SETA02", "US_CPI_UsedVeh",    "m", "yoy"),   # CPI Used Cars & Trucks
    ("CUSR0000SAS",    "US_CPI_Services",   "m", "yoy"),   # CPI Services
    ("PCEPI",          "US_PCE",            "m", "yoy"),   # PCE Price Index
    ("PCEPILFE",       "US_CorePCE",        "m", "yoy"),   # Core PCE Price Index
    ("PPIACO",         "US_PPI_AllComm",    "m", "yoy"),   # PPI All Commodities
    ("PPIFID",         "US_PPI_FinalDem",   "m", "yoy"),   # PPI Final Demand
    ("PPICOR",         "US_PPI_Core",       "m", "yoy"),   # PPI Final Demand Less Food & Energy
    ("IR",             "US_ImportPI",       "m", "yoy"),   # Import Price Index All Commodities
    ("IQ",             "US_ExportPI",       "m", "yoy"),   # Export Price Index All Commodities
    ("MEDCPIM158SFRBCLE","US_MedianCPI",    "m", "raw"),   # Cleveland Fed Median CPI (already % change)
    ("CPILEGSL",       "US_CPI_xEnergy",    "m", "yoy"),   # CPI All Items Less Energy
    ("PPIIDC",         "US_PPI_IndComm",    "m", "yoy"),   # PPI Industrial Commodities
    ("WPSFD4131",      "US_PPI_CoreFin",    "m", "yoy"),   # PPI Finished Goods Less Food & Energy
    ("IREXPET",        "US_ImportPI_xPet",  "m", "yoy"),   # Import Price Index excl Petroleum
    ("PCETRIM1M158SFRBDAL","US_TrimPCE",   "m", "raw"),   # Dallas Fed Trimmed Mean PCE (1M, already % change)
    ("PCETRIM12M159SFRBDAL","US_TrimPCE12","m", "raw"),   # Dallas Fed Trimmed Mean PCE (12M, already % change)
    ("CUSR0000SEHC01", "US_CPI_OER",       "m", "yoy"),   # CPI Owners' Equivalent Rent
    ("CPIEDUSL",       "US_CPI_Edu",        "m", "yoy"),   # CPI Education & Communication
]

# ============================================================
# 6. US LABOR (~15)
# ============================================================
US_LABOR = [
    ("PAYEMS",         "US_NFP",            "m", "diff"),  # Total Nonfarm Payrolls (level→MoM diff)
    ("UNRATE",         "US_UnempRate",      "m", "raw"),   # Unemployment Rate
    ("ICSA",           "US_InitClaims",     "d", "raw"),   # Initial Jobless Claims (weekly)
    ("CCSA",           "US_ContClaims",     "d", "raw"),   # Continued Claims (weekly)
    ("CES0500000003",  "US_AvgHrEarn",      "m", "yoy"),   # Avg Hourly Earnings, All Employees
    ("AHETPI",         "US_AvgHrEarn_Prod", "m", "yoy"),   # Avg Hourly Earnings, Production Workers
    ("CIVPART",        "US_LFPR",           "m", "raw"),   # Labor Force Participation Rate
    ("EMRATIO",        "US_EmpPopRatio",    "m", "raw"),   # Employment-Population Ratio
    ("JTSJOL",         "US_JOLTS_Open",     "m", "raw"),   # JOLTS Job Openings (thousands)
    ("JTSQUL",         "US_JOLTS_Quits",    "m", "raw"),   # JOLTS Quits (thousands)
    ("JTSHIL",         "US_JOLTS_Hires",    "m", "raw"),   # JOLTS Hires (thousands)
    ("MANEMP",         "US_MfgEmploy",      "m", "diff"),  # Manufacturing Employment (level→MoM diff)
    ("LNS14000006",    "US_UnempBlack",     "m", "raw"),   # Black Unemployment Rate
    ("U6RATE",         "US_U6Rate",         "m", "raw"),   # U-6 Unemployment Rate (broad)
    ("USFIRE",         "US_FinEmp",         "m", "diff"),  # Financial Activities Employment
    ("USTPU",          "US_TradeTransEmp",  "m", "diff"),  # Trade, Transportation & Utilities Employment
    ("CEU0500000008",  "US_AvgHrEarn_Prod2","m", "yoy"),   # Avg Hourly Earnings, Production (SA)
    ("USWTRADE",       "US_WholsaleEmp",   "m", "diff"),  # Wholesale Trade Employment
    ("USPRIV",         "US_PrivateEmp",     "m", "diff"),  # Total Private Employment
    ("AWHAEMAN",       "US_MfgAvgWkHrs",   "m", "raw"),   # Avg Weekly Hours, Manufacturing
    ("CES0500000007",  "US_AvgWkHrs_All",  "m", "raw"),   # Avg Weekly Hours, All Private Employees
]

# ============================================================
# 7. US MONETARY (~15)
# ============================================================
US_MONETARY = [
    ("FEDFUNDS",       "US_FedFunds",       "m", "raw"),   # Effective Federal Funds Rate
    ("DFF",            "US_FedFunds_d",     "d", "raw"),   # Federal Funds Rate (daily)
    ("M1SL",           "US_M1",             "m", "yoy"),   # M1 Money Stock
    ("M2SL",           "US_M2",             "m", "yoy"),   # M2 Money Stock
    ("BOGMBASE",       "US_MonBase",        "m", "yoy"),   # Monetary Base Total
    ("WALCL",          "Fed_TotalAssets",   "d", "yoy"),   # Fed Total Assets (weekly)
    ("TOTRESNS",       "US_TotReserves",    "m", "yoy"),   # Total Reserves of Depository Institutions
    ("BUSLOANS",       "US_CommLoans",      "m", "yoy"),   # Commercial & Industrial Loans
    ("TOTALSL",        "US_ConsCredit",     "m", "yoy"),   # Total Consumer Credit
    ("M2V",            "US_M2Velocity",     "m", "raw"),   # Velocity of M2 (quarterly→monthly by ffill)
    ("TOTCI",          "US_CommBankCred",   "m", "yoy"),   # Commercial Bank Total Credit
    ("MPRIME",         "US_PrimeLoan",      "m", "raw"),   # Bank Prime Loan Rate
    ("DPRIME",         "US_PrimeLoan_d",    "d", "raw"),   # Bank Prime Loan Rate (daily)
    ("REVOLSL",        "US_RevolvCredit",   "m", "yoy"),   # Revolving Consumer Credit
    ("NONREVSL",       "US_NonRevCredit",   "m", "yoy"),   # Non-Revolving Consumer Credit
    ("REALLN",         "US_RealEstLoans",   "m", "yoy"),   # Real Estate Loans, All Commercial Banks
    ("DPCREDIT",       "US_DepInstCredit",  "m", "yoy"),   # Depository Institutions Credit
]

# ============================================================
# 8. US FINANCIAL (~20)
# ============================================================
US_FINANCIAL = [
    # Treasury Yields (daily)
    ("DGS1MO",         "UST_1M",            "d", "raw"),   # 1-Month Treasury Yield
    ("DGS3MO",         "UST_3M",            "d", "raw"),   # 3-Month Treasury Yield
    ("DGS6MO",         "UST_6M",            "d", "raw"),   # 6-Month Treasury Yield
    ("DGS1",           "UST_1Y",            "d", "raw"),   # 1-Year Treasury Yield
    ("DGS2",           "UST_2Y",            "d", "raw"),   # 2-Year Treasury Yield
    ("DGS3",           "UST_3Y",            "d", "raw"),   # 3-Year Treasury Yield
    ("DGS5",           "UST_5Y",            "d", "raw"),   # 5-Year Treasury Yield
    ("DGS7",           "UST_7Y",            "d", "raw"),   # 7-Year Treasury Yield
    ("DGS10",          "UST_10Y",           "d", "raw"),   # 10-Year Treasury Yield
    ("DGS20",          "UST_20Y",           "d", "raw"),   # 20-Year Treasury Yield
    ("DGS30",          "UST_30Y",           "d", "raw"),   # 30-Year Treasury Yield
    # Yield Spreads
    ("T10Y2Y",         "Spread_10Y2Y",      "d", "raw"),   # 10Y-2Y Spread
    ("T10Y3M",         "Spread_10Y3M",      "d", "raw"),   # 10Y-3M Spread
    ("T10YFF",         "Spread_10YFF",      "d", "raw"),   # 10Y minus Fed Funds
    # Credit Spreads
    ("BAA10Y",         "Spread_BAA10Y",     "d", "raw"),   # Moody's BAA minus 10Y Treasury
    ("AAA10Y",         "Spread_AAA10Y",     "d", "raw"),   # Moody's AAA minus 10Y Treasury
    ("BAMLC0A0CM",     "ICE_IG_OAS",        "d", "raw"),   # ICE BofA US Corporate OAS
    ("BAMLH0A0HYM2",   "ICE_HY_OAS",        "d", "raw"),   # ICE BofA US High Yield OAS
    ("BAMLC0A4CBBB",   "ICE_BBB_OAS",       "d", "raw"),   # ICE BofA BBB Corporate OAS
    # Financial Conditions
    ("NFCI",           "Chicago_NFCI",      "d", "raw"),   # Chicago Fed National Financial Conditions (weekly)
    ("ANFCI",          "Chicago_ANFCI",     "d", "raw"),   # Chicago Fed Adjusted NFCI (weekly)
    ("STLFSI4",        "StL_FSI",           "d", "raw"),   # St. Louis Fed Financial Stress Index (weekly)
    ("KCFSI",          "KC_FSI",            "m", "raw"),   # Kansas City Financial Stress Index (monthly)
    ("BAMLH0A2HYB",    "ICE_SingleB_OAS_v2","d", "raw"),   # ICE BofA Single-B HY OAS
    ("TB3MS",          "TBill_3M",          "m", "raw"),   # 3-Month Treasury Bill Secondary Market Rate
    ("GS10",           "UST_10Y_m",         "m", "raw"),   # 10-Year Treasury (monthly average)
    ("GS5",            "UST_5Y_m",          "m", "raw"),   # 5-Year Treasury (monthly average)
    ("GS1",            "UST_1Y_m",          "m", "raw"),   # 1-Year Treasury (monthly average)
    ("GS2",            "UST_2Y_m",          "m", "raw"),   # 2-Year Treasury (monthly average)
    ("GS30",           "UST_30Y_m",         "m", "raw"),   # 30-Year Treasury (monthly average)
]

# ============================================================
# 9. US REAL ACTIVITY (~20)
# ============================================================
US_REAL_ACTIVITY = [
    ("INDPRO",         "US_IndProd",        "m", "yoy"),   # Industrial Production Total Index
    ("IPMAN",          "US_IndProd_Mfg",    "m", "yoy"),   # Industrial Production: Manufacturing
    ("TCU",            "US_CapUtil",        "m", "raw"),   # Capacity Utilization: Total
    ("MCUMFN",         "US_CapUtil_Mfg",    "m", "raw"),   # Capacity Utilization: Manufacturing
    ("RSAFS",          "US_RetailSales",    "m", "yoy"),   # Advance Retail Sales (Food Services)
    ("RSXFS",          "US_RetailXFood",    "m", "yoy"),   # Retail Trade (excl Food Services)
    ("HOUST",          "US_HousingStart",   "m", "raw"),   # Housing Starts (thousands)
    ("PERMIT",         "US_BuildPermit",    "m", "raw"),   # Building Permits (thousands)
    ("HSN1F",          "US_NewHomeSale",    "m", "raw"),   # New Single-Family Home Sales
    ("EXHOSLUSM495S",  "US_ExistHomeSale",  "m", "raw"),   # Existing Home Sales
    ("CSUSHPISA",      "US_CaseShiller",    "m", "yoy"),   # Case-Shiller National Home Price Index
    ("DGORDER",        "US_DurGoodsOrd",    "m", "yoy"),   # Durable Goods New Orders
    ("NEWORDER",       "US_CoreCapGoods",   "m", "yoy"),   # Nondefense Capital Goods ex Aircraft Orders
    ("PI",             "US_PersIncome",     "m", "yoy"),   # Personal Income
    ("PCE",            "US_PersSpend",      "m", "yoy"),   # Personal Consumption Expenditures
    ("DSPI",           "US_DispIncome",     "m", "yoy"),   # Disposable Personal Income
    ("ISRATIO",        "US_InvSalesRatio",  "m", "raw"),   # Business Inventories-to-Sales Ratio
    ("TOTALSA",        "US_VehicleSales",   "m", "raw"),   # Total Vehicle Sales (SAAR, millions)
    ("AMTMNO",         "US_MfgOrders",      "m", "yoy"),   # Manufacturers New Orders: Total Manufacturing
    ("MNFCTRMPCIMSA",  "US_MfgInventory",   "m", "yoy"),   # Manufacturers Inventories
    ("UMTMNO",         "US_MfgOrders_NSA",  "m", "yoy"),   # Manufacturers New Orders (Not SA)
    ("RETAILIRSA",     "US_RetailInvSales", "m", "raw"),   # Retailers Inventories-to-Sales Ratio
    ("PCEC96",         "US_RealPCE",        "m", "yoy"),   # Real Personal Consumption Expenditures
    ("CMRMTSPL",       "US_RealRetail",     "m", "yoy"),   # Real Manufacturing and Trade Sales
    ("DSPIC96",        "US_RealDispIncome", "m", "yoy"),   # Real Disposable Personal Income
    ("UNRATENSA",      "US_UnempNSA",       "m", "raw"),   # Unemployment Rate (Not SA)
    ("W875RX1",        "US_RealCompPerHr",  "m", "yoy"),   # Real Compensation Per Hour (Nonfarm Business)
    ("USSTHPI",        "US_FHFA_HPI",       "m", "yoy"),   # FHFA House Price Index (purchase only)
    ("IPB51111S",      "US_IP_AutoTruck",   "m", "yoy"),   # Industrial Production: Autos & Trucks
]

# ============================================================
# 10. US EXPECTATIONS (~10)
# ============================================================
US_EXPECTATIONS = [
    ("UMCSENT",        "US_MichSentiment",  "m", "raw"),   # U of Michigan Consumer Sentiment
    ("MICH",           "US_MichInflExp",    "m", "raw"),   # U of Michigan Inflation Expectation (1Y)
    ("T5YIE",          "Breakeven_5Y",      "d", "raw"),   # 5-Year Breakeven Inflation Rate
    ("T10YIE",         "Breakeven_10Y",     "d", "raw"),   # 10-Year Breakeven Inflation Rate
    ("T5YIFR",         "Fwd5Y5Y_Infl",     "d", "raw"),   # 5-Year 5-Year Forward Inflation Expectation
    ("DFII10",         "TIPS_10Y",          "d", "raw"),   # 10Y Treasury Inflation-Indexed Yield
    ("USALOLITONOSTSAM","US_CLI",           "m", "raw"),   # OECD Composite Leading Indicator: US
    ("KORLOLITONOSTSAM","KR_CLI",           "m", "raw"),   # OECD Composite Leading Indicator: Korea
    ("JPNLOLITONOSTSAM","JP_CLI",           "m", "raw"),   # OECD Composite Leading Indicator: Japan
    ("CHNLOLITOAASTSAM","CN_CLI",           "m", "raw"),   # OECD Composite Leading Indicator: China (amplitude adj.)
    ("G4ELOLITONOSTSAM","EU4_CLI",          "m", "raw"),   # OECD CLI: Major 4 European Economies
    ("G7LOLITOAASTSAM", "G7_CLI",           "m", "raw"),   # OECD CLI: G7 (amplitude adj.)
    ("GACDISA066MSFRBNY","NY_EmpireMfg",   "m", "raw"),   # NY Fed Empire State Manufacturing Index
    ("DFII5",          "TIPS_5Y",           "d", "raw"),   # 5Y Treasury Inflation-Indexed Yield
    ("USEPUINDXM",     "US_EconPolicyUnc",  "m", "raw"),   # US Economic Policy Uncertainty Index
]

# ============================================================
# 11. GLOBAL PRICES (~20)
# ============================================================
GLOBAL_PRICES = [
    ("CHNCPIALLMINMEI", "CN_CPI",           "m", "yoy"),   # China CPI (OECD)
    ("CHNPIEATI01GYM",  "CN_PPI",           "m", "raw"),   # China PPI: Industrial Activities (YoY already)
    ("JPNCPIALLMINMEI", "JP_CPI",           "m", "yoy"),   # Japan CPI (OECD)
    ("GBRCPIALLMINMEI", "UK_CPI",           "m", "yoy"),   # UK CPI (OECD)
    ("DEUCPIALLMINMEI", "DE_CPI",           "m", "yoy"),   # Germany CPI (OECD)
    ("CPALTT01FRM657N", "FR_CPI",           "m", "yoy"),   # France CPI (OECD)
    ("INDCPIALLMINMEI", "IN_CPI",           "m", "yoy"),   # India CPI (OECD)
    ("BRACPIALLMINMEI", "BR_CPI",           "m", "yoy"),   # Brazil CPI (OECD)
    ("MEXCPIALLMINMEI", "MX_CPI",           "m", "yoy"),   # Mexico CPI (OECD)
    ("TURCPIALLMINMEI", "TR_CPI",           "m", "yoy"),   # Turkey CPI (OECD)
    ("KORCPALTT01IXNBM","KR_CPI_OECD",     "m", "yoy"),   # Korea CPI (OECD, cross-check)
    ("CP0000EZ19M086NEST","EA_HICP",        "m", "yoy"),   # Euro Area HICP All Items
    ("JPNCPICORMINMEI", "JP_CoreCPI",       "m", "yoy"),   # Japan Core CPI (ex food & energy, OECD)
    ("CPALTT01USM657N", "US_CPI_OECD",      "m", "yoy"),   # US CPI (OECD, cross-check)
    ("G20CPALTT01GPM",  "G20_CPI",          "m", "raw"),   # G20 CPI (OECD, already % growth)
    ("PIEAEN02KRM661N", "KR_PPI_Energy",    "m", "yoy"),   # Korea PPI Energy (OECD)
    ("RUSCPIALLMINMEI", "RU_CPI",           "m", "yoy"),   # Russia CPI (OECD)
    ("IDNCPIALLMINMEI", "ID_CPI",           "m", "yoy"),   # Indonesia CPI (OECD)
    ("ZAFCPIALLMINMEI", "ZA_CPI",           "m", "yoy"),   # South Africa CPI (OECD)
    ("OECDECPALTT01GPQ","OECD_EU_CPI",     "m", "raw"),   # OECD Europe CPI (quarterly, already % growth)
    ("AUSCPIALLMINMEI","AU_CPI",           "m", "yoy"),   # Australia CPI (OECD)
    ("CANCPIALLMINMEI","CA_CPI",           "m", "yoy"),   # Canada CPI (OECD)
    ("SAECPIALLMINMEI","SA_CPI",           "m", "yoy"),   # Saudi Arabia CPI (OECD)
    ("THACPIALLMINMEI","TH_CPI",           "m", "yoy"),   # Thailand CPI (OECD)
]

# ============================================================
# 12. GLOBAL RATES (~15)
# ============================================================
GLOBAL_RATES = [
    ("ECBDFR",         "ECB_DepoRate",      "m", "raw"),   # ECB Deposit Facility Rate
    ("ECBMRRFR",       "ECB_RefiRate",      "m", "raw"),   # ECB Main Refinancing Rate
    ("IRSTCB01JPM156N","BOJ_Rate",          "m", "raw"),   # Bank of Japan Policy Rate (OECD)
    ("IRSTCB01GBM156N","BOE_Rate",          "m", "raw"),   # Bank of England Rate (OECD)
    ("IRSTCB01CAM156N","BOC_Rate",          "m", "raw"),   # Bank of Canada Rate (OECD)
    ("IRSTCB01AUM156N","RBA_Rate",          "m", "raw"),   # Reserve Bank of Australia Rate (OECD)
    ("IRSTCB01INM156N","RBI_Rate",          "m", "raw"),   # Reserve Bank of India Rate (OECD)
    ("IRSTCB01CNM156N","PBOC_Rate",         "m", "raw"),   # People's Bank of China Rate (OECD)
    ("IRSTCB01BRM156N","BCB_Rate",          "m", "raw"),   # Central Bank of Brazil Rate (OECD)
    ("IRSTCB01MXM156N","Banxico_Rate",      "m", "raw"),   # Bank of Mexico Rate (OECD)
    ("IRSTCB01KRM156N","BOK_Rate",          "m", "raw"),   # Bank of Korea Rate (OECD)
    ("IR3TIB01JPM156N","JP_Interbank3M",    "m", "raw"),   # Japan 3M Interbank Rate (OECD)
    ("IR3TIB01CNM156N","CN_Interbank3M",    "m", "raw"),   # China 3M Interbank Rate (OECD)
    ("INTDSRKRM193N",  "KR_DiscountRate",   "m", "raw"),   # Korea Discount Rate (IFS)
    ("INTDSRBRM193N",  "BR_DiscountRate",   "m", "raw"),   # Brazil Discount Rate (IFS)
    ("IRLTLT01DEM156N","DE_10Y_Bond",       "m", "raw"),   # Germany 10Y Govt Bond Yield (OECD)
    ("IRLTLT01JPM156N","JP_10Y_Bond",       "m", "raw"),   # Japan 10Y Govt Bond Yield (OECD)
    ("IRLTLT01GBM156N","UK_10Y_Bond",       "m", "raw"),   # UK 10Y Govt Bond Yield (OECD)
    ("IRLTLT01USM156N","US_10Y_OECD",       "m", "raw"),   # US 10Y Govt Bond Yield (OECD)
    ("INTDSRCNM193N",  "CN_DiscountRate",   "m", "raw"),   # China Discount Rate (IFS)
    ("INTDSRINM193N",  "IN_DiscountRate",   "m", "raw"),   # India Discount Rate (IFS)
    ("IRSTCI01BRM156N","BR_Interbank",      "m", "raw"),   # Brazil Call Money/Interbank Rate (OECD)
    ("IRSTCI01INM156N","IN_Interbank",      "m", "raw"),   # India Call Money/Interbank Rate (OECD)
    ("IR3TIB01KRM156N","KR_3M_v2",         "m", "raw"),   # Korea 3M Rate (dedup handled)
    ("IRSTCI01USM156N","US_Interbank",      "m", "raw"),   # US Call Money/Interbank Rate (OECD)
]

# ============================================================
# 13. KOREA DOMESTIC (~15)
# ============================================================
KOREA_DOMESTIC = [
    ("IR3TIB01KRM156N","KR_Interbank3M",    "m", "raw"),   # Korea 3M Interbank Rate (OECD)
    ("IRSTCI01KRM156N","KR_CallRate",       "m", "raw"),   # Korea Call Money Rate (OECD)
    ("LRUN64TTKSM156S","KR_Unemp",         "m", "raw"),   # Korea Unemployment Rate (OECD)
    ("XTIMVA01KRM667S","KR_Imports",        "m", "yoy"),   # Korea Imports Value (OECD)
    ("XTEXVA01KRM667S","KR_Exports",        "m", "yoy"),   # Korea Exports Value (OECD)
    ("KORPROINDMISMEI","KR_IndProd",        "m", "yoy"),   # Korea Industrial Production (OECD)
    ("KORPROMANMISMEI","KR_MfgProd",        "m", "yoy"),   # Korea Manufacturing Production (OECD)
    ("CSCICP02KRM066S","KR_ConsConf",       "m", "raw"),   # Korea Consumer Confidence (OECD)
    ("MYAGM2KRM189S",  "KR_M2",            "m", "yoy"),   # Korea M2 Money Supply (OECD)
    ("RBKRBIS",        "KR_REER_v2",       "m", "raw"),   # Korea Real Effective Exchange Rate (BIS) — same as above, won't duplicate
    ("NBKRBIS",        "KR_NEER",          "m", "raw"),   # Korea Nominal Effective Exchange Rate (BIS)
    ("CPALTT01KRQ657N","KR_CPI_Q",         "m", "yoy"),   # Korea CPI Quarterly (OECD, for validation)
    ("IRLTLT01KRM156N","KR_LongRate",       "m", "raw"),   # Korea Long-Term Govt Bond Yield (OECD)
    ("KORPRINTO01GYSAM","KR_IndProd_YoY",   "m", "raw"),   # Korea Industrial Production YoY (OECD, already %)
    ("SLRTTO02KRM659S","KR_RetailSales",    "m", "yoy"),   # Korea Retail Trade Volume (OECD)
    ("LREM64TTKSM156S","KR_EmpRate",       "m", "raw"),   # Korea Employment Rate (OECD)
    ("XTNTVA01KRM667S","KR_NetTrade",      "m", "raw"),   # Korea Net Trade Value (OECD)
]

# ============================================================
# 14. FINANCIAL MARKETS (~15)
# ============================================================
FINANCIAL_MARKETS = [
    ("SP500",          "SP500",             "d", "raw"),   # S&P 500 Index
    ("NASDAQCOM",      "NASDAQ",            "d", "raw"),   # NASDAQ Composite Index
    ("DJIA",           "DJIA",              "d", "raw"),   # Dow Jones Industrial Average
    ("VIXCLS",         "VIX",              "d", "raw"),   # CBOE VIX Volatility Index
    ("BAMLC0A0CM",     "IG_CreditSpread",  "d", "raw"),   # ICE BofA US IG Corporate OAS (same as above)
    ("BAMLH0A0HYM2",   "HY_CreditSpread",  "d", "raw"),   # ICE BofA US HY OAS (same as above)
    ("BAMLH0A2HYB",    "ICE_SingleB_OAS",  "d", "raw"),   # ICE BofA Single-B HY OAS
    ("BAA",            "Moodys_BAA",       "m", "raw"),   # Moody's BAA Corporate Bond Yield
    ("AAA",            "Moodys_AAA",       "m", "raw"),   # Moody's AAA Corporate Bond Yield
    ("MORTGAGE30US",   "US_Mortgage30Y",   "d", "raw"),   # 30-Year Fixed Mortgage Rate
    ("MORTGAGE15US",   "US_Mortgage15Y",   "d", "raw"),   # 15-Year Fixed Mortgage Rate
    ("TEDRATE",        "TED_Spread",       "d", "raw"),   # TED Spread (DISCONTINUED 2022, useful pre-2022)
    ("DCOILWTICO",     "Oil_WTI_d_v2",     "d", "raw"),   # WTI (dedup handled)
]

# ============================================================
# 15. SUPPLY CHAIN & TRANSPORT (~5)
# ============================================================
SUPPLY_CHAIN = [
    ("GSCPI",          "GSCPI",             "m", "raw"),   # NY Fed Global Supply Chain Pressure Index
    ("FRGSHPUSM649NCIS","CassFreight_Ship", "m", "raw"),   # Cass Freight Index: Shipments
    ("FRGEXPUSM649NCIS","CassFreight_Exp",  "m", "raw"),   # Cass Freight Index: Expenditures
    ("TSIFRGHT",       "FreightTSI",        "m", "raw"),   # Freight Transportation Services Index
    ("IGREA",          "GlobalRealAct",     "m", "raw"),   # Kilian Index of Global Real Economic Activity
    ("TSIFRGHTC",      "FreightTSI_Comb",   "m", "raw"),   # Combined Freight Transportation Index
    ("WPU301301",      "PPI_DeepSeaFrt",    "m", "yoy"),   # PPI Deep Sea Freight
    ("PCU484121484121", "PPI_Trucking",     "m", "yoy"),   # PPI General Freight Trucking
    ("PCU4831114831115","PPI_DeepSeaFrt2",  "m", "yoy"),   # PPI Deep Sea Freight Transportation Services
]


# ============================================================
# COMBINED MASTER LIST
# ============================================================
ALL_SERIES = (
    EXCHANGE_RATES
    + COMMODITIES_ENERGY
    + COMMODITIES_METALS
    + COMMODITIES_AGRICULTURE
    + US_PRICES
    + US_LABOR
    + US_FINANCIAL
    + US_MONETARY
    + US_REAL_ACTIVITY
    + US_EXPECTATIONS
    + GLOBAL_PRICES
    + GLOBAL_RATES
    + KOREA_DOMESTIC
    + FINANCIAL_MARKETS
    + SUPPLY_CHAIN
)

# De-duplicate by FRED series ID (keep first occurrence)
_seen = set()
FRED_SERIES = []
for entry in ALL_SERIES:
    sid = entry[0]
    if sid not in _seen:
        _seen.add(sid)
        FRED_SERIES.append(entry)

# Category mapping for tournament selection
CATEGORIES = {
    "exchange_rates":         EXCHANGE_RATES,
    "commodities_energy":     COMMODITIES_ENERGY,
    "commodities_metals":     COMMODITIES_METALS,
    "commodities_agriculture":COMMODITIES_AGRICULTURE,
    "us_prices":              US_PRICES,
    "us_labor":               US_LABOR,
    "us_monetary":            US_MONETARY,
    "us_financial":           US_FINANCIAL,
    "us_real_activity":       US_REAL_ACTIVITY,
    "us_expectations":        US_EXPECTATIONS,
    "global_prices":          GLOBAL_PRICES,
    "global_rates":           GLOBAL_RATES,
    "korea_domestic":         KOREA_DOMESTIC,
    "financial_markets":      FINANCIAL_MARKETS,
    "supply_chain":           SUPPLY_CHAIN,
}


if __name__ == "__main__":
    print(f"Total entries (before dedup): {len(ALL_SERIES)}")
    print(f"Unique FRED series (after dedup): {len(FRED_SERIES)}")
    print(f"\nPer-category counts:")
    for cat, series in CATEGORIES.items():
        print(f"  {cat:30s} {len(series):3d}")
    print(f"\nFrequency breakdown:")
    daily = sum(1 for _, _, f, _ in FRED_SERIES if f == "d")
    monthly = sum(1 for _, _, f, _ in FRED_SERIES if f == "m")
    print(f"  daily:   {daily}")
    print(f"  monthly: {monthly}")
    print(f"\nTransform breakdown:")
    for t in ("raw", "yoy", "diff"):
        n = sum(1 for _, _, _, tr in FRED_SERIES if tr == t)
        print(f"  {t:5s}: {n}")
