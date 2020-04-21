import os
import pandas as pd
from collections import Counter

hw_folder = ['fid0', 'fid1', 'fid2', 'fid3']

def in_range(conditions, u_in):
	try:
		x = float(conditions[0])
		y = float(conditions[1])
		u_in = float(u_in)
		if x>=y:
			return y <= u_in <= x
		else:
			return y >= u_in >= x
	except:
		return False

outsheet = pd.DataFrame()
grade_sheet = pd.read_excel('Midterm_template_KEY_DG3.xlsx', sheet_name='Answer_Sheet')
grade_sheet = grade_sheet.fillna('NaN')
students = []
grades = []
groups = []
excel_sheet = []
incorrects_list = []
scores = []
incorrect_total = []
for group in hw_folder:
	hws = os.listdir(group)
	for i in hws:

		try:

			print(i)
			user_answer = pd.read_excel('{}/{}'.format(group, i), sheet_name='Answer_Sheet')
			terms = i.split('_')
			name = terms[0] + ' '
			if '8' in terms[2] or "6" in terms[2]:
				if '8' in terms[2]:
					name += terms[1] + ' ' + terms[2].split('8')[0]
				else:
					name += terms[1] + ' ' + terms[2].split('8')[0]
			else:
				if '8' in terms[1]:
					name += terms[1].split('8')[0]
				else:
					name += terms[1].split('8')[0]


			group = group
			late=False
			if 'late' in terms:
				late=True
			correct = 0
			score = 0
			incorrects = []
			for dex in user_answer.index:
				submitted_answer = user_answer['Answer'][dex]
				eval_type = grade_sheet['Type'][dex]
				if eval_type == 'range':
					if in_range([grade_sheet['con1'][dex], grade_sheet['con2'][dex]], submitted_answer) == True:
						correct +=1
						score+=grade_sheet['Points'][dex]
					else:
						incorrects.append(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						print(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						incorrect_total.append(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						
				elif eval_type == 'exact':

					if grade_sheet['Answer'][dex] == submitted_answer:
						correct+=1
						score+=grade_sheet['Points'][dex]
					else:
						incorrects.append(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						print(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						incorrect_total.append(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])

				elif eval_type == 'ranges':
					cons = ['con1', 'con2', 'con3', 'con4', 'con5', 'con6', 'con7']
					res=False
					for x in cons:
						if grade_sheet[x][dex] != 'NaN':
							if in_range(eval(grade_sheet[x][dex]), submitted_answer) == True:
								res=True
						else:
							pass
					if res==True:
						correct+=1
						score+=grade_sheet['Points'][dex]
					else:
						incorrects.append(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						print(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
						incorrect_total.append(user_answer['Question Sheet'][dex] + ' - ' + user_answer['Part'][dex])
				elif eval_type == 'super_range_shy_bayesian':
					correct+=1
					score+=grade_sheet['Points'][dex]

			print('correct: ' + str(correct))
			students.append(name)
			grades.append(correct)
			groups.append(group)
			excel_sheet.append('C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts/Grading_test/{}/{}'.format(group, i))
			incorrects_list.append(incorrects)
			scores.append(score)

		except:
			terms = i.split('_')
			print(i + ' broken')
			name = terms[0]
			grade = 'broken'
			students.append(name)
			grades.append(grade)
			groups.append(group)
			excel_sheet.append('C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts/Grading_test/{}/{}'.format(group, i))
			incorrects_list.append(incorrects)
			scores.append(score)

print(Counter(incorrect_total))

outsheet['student'] = students
outsheet['grade'] = grades
outsheet['score'] = scores
outsheet['incorrect'] = incorrects_list
outsheet['group'] = groups
outsheet['link'] = excel_sheet

outsheet.to_csv('grades_final.csv')