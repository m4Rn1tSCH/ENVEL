import QuantLib as ql
import pandas as pd


def bond_PV(annual_rate,coupon_rate, day_count, comp_type, frequency, issue_date, maturity_date, tenor, calendar, business_convention,
			date_gen, month_end, face_value, settlement_days, discount_const):
	#SET UP INTEREST RATE FOR BOND
	#Bond 1
	annual_rate = 0.075
	day_count = ql.ActualActual()
	comp_type = ql.Compounded
	frequency = ql.Semiannual
	interest_rate = ql.InterestRate(annualRate, day_count, comp_type, frequency)
	
	#SET UP COUPON SCHEDULE FOR BOND
	#Bond 1
	coupon_rate = .06
	coupons = [coupon_rate]
	
	#SET UP ISSUE, MATURITY DATES and SCHEDULES FOR BOND
	#Bond 1
	issue_date = ql.Date(1, 4, 2018)
	maturity_date = ql.Date(1, 4, 2023) #SET THIS
	tenor = ql.Period(ql.Semiannual) #SET THIS
	calendar = ql.UnitedStates()
	business_convention = ql.Unadjusted
	date_gen = ql.DateGeneration.Backward
	month_end = True		#NEEDED FOR ANNUITY OR ANNUITY DUE COUPON PAYMENTS ONLY
	schedule = ql.Schedule(issue_date, maturity_date, tenor, calendar, business_convention,
								business_convention , date_gen, month_end)
	
	
	# build the fixed rate components for the bond
	settlement_days = 0
	face_value = 1000
	fixed_rate_bond = ql.FixedRateBond(settlement_days, face_value, schedule, coupons, day_count)
	fixed_rate_bond_2 = ql.FixedRateBond(settlement_days, face_value, schedule_2, coupons_2, day_count)
	
	
	#SET THE MARKET YIELD OF COMPARABLE INVESTMENTS
	discount_const = 0.0725
	
	# create a bond engine with a fixed discount rate and use this bond engine;
	bond_engine_const = ql.DiscountingBondEngine(discount_const)
	fixed_rate_bond.setPricingEngine(bond_engine_const)
	
	
	# Calculate the price of a single bond
	bond_price = fixed_rate_bond.NPV()
	df_schedule = pd.DataFrame(schedule)
	print(bond_price)
	print(df_schedule)


