

def ac_entry(start,end,label):
	return {
		'start':start,
		'end':end,
		'label': label
	}

def ac_line(entry):
	a=float(entry['start'])
	b=float(entry['end'])
	c=entry['label']
	return "{:5.6f}\t{:5.6f}\t{}\n".format(a,b,c)


def save_ac_output(filename, label_list):
	with open(filename, "wt") as outfile:
		for entry in label_list:
			label = ac_line(entry)
			outfile.write(label)

