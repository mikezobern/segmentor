from get_batch import Train_data
import torch
td = Train_data(6)



def pairs_distance():

    el, tl = td.get_bunch_of_embs_by_row_id()
    elt = torch.tensor(data=el)
    tl = torch.tensor(data=tl)


    s_to_not_relatives = 0
    s_to_relative = 0
    relative_index = torch.argmax(tl)
    if relative_index==0:
        s_to_relative = None
        s_to_not_relatives = None
        '''
        for i in range(1, 6):
            s_to_not_relatives+=torch.sum(elt[0]*elt[i])
            s_to_not_relatives = float(s_to_not_relatives)'''
    else:
        for i in range(1,6):
            if i == relative_index:
                s_to_relative = torch.sum(elt[0]*elt[i])
                s_to_relative = float(s_to_relative)
            else:
                s_to_not_relatives += torch.sum(elt[0]*elt[i])
                s_to_not_relatives = float(s_to_not_relatives)
        s_to_not_relatives = s_to_not_relatives/4
    return s_to_relative, s_to_not_relatives

lst_rel = []
lst_not_rel = []
for i in range(200):
    r, n = pairs_distance()
    if r!=None:
        lst_rel+=[r]
    if n!=None:
        lst_not_rel+=[n]

print('lst_rel', sum(lst_rel)/len(lst_rel))
print('lst_not_rel', sum(lst_not_rel)/len(lst_not_rel))