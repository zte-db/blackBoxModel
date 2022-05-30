data = {'1':{"a":1},"5":"b","2":"c"}
_data = sorted(list(data.items()),key=lambda item:item[0])
print(_data)
