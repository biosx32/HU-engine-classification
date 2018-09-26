import Helpers


train_data = Helpers.get_training_data('ManualClassify.csv')
batch_size = 100
batch_count = len(train_data)/batch_size
max_count = len(train_data)


def get_pair(train_data):
    result = {'label': [x['label'] for x in train_data],
              'data': [x['data'] for x in train_data]}
    return result


j = 0
def get_next():
    global j
    p=j+batch_size
    if p > max_count:
        p = max_count
    hundred = train_data[j:p]
    j+=batch_size

    result = {'label': [x['label'] for x in hundred],
              'data': [x['data'] for x in hundred]}
    return result

print(get_next())
print(get_next())
print(get_next())
print(get_next())
print(get_next())
print(get_next())
