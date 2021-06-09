import os 


base = '/raid/celong/FaceScape/fsmview_images'
newbase = '/raid/celong/FaceScape/jsons'
if not os.path.exists(  newbase):
        os.mkdir( newbase )
for id in os.listdir(base):
    if not os.path.exists(  newbase + '/' + id ):
        os.mkdir( newbase + '/' + id)
    for exp in os.listdir( base + '/' + id ):
        if not os.path.exists(  newbase + '/' + id  + '/' + exp):
            os.mkdir( newbase + '/' + id + '/' + exp)
        command ='cp  ' +  newbase + '/' + id  + '/' + exp +'params.json' + ' ' + newbase + '/' + id + '/' + exp
        os.system(command)
