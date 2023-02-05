import ast

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def get_tags(example):
    example.TokenList=ast.literal_eval(example.TokenList)
    sen_words=example.TokenList
    L2=str(example.pattern).split()
    Ltot5=[]
    k=0
    for i in range(0,len(sen_words)):
        tagf=L2[i]    
        Lp=sen_words[i].split()
        if len(Lp)>1:
            tag='B-'+tagf
            Ltot5.insert(k,tag)
            k=k+1        
            for j in range(1,len(Lp)):
                tag='I-'+tagf 
                Ltot5.insert(k,tag)
                k=k+1    
        else:
            tag='B-'+tagf
            Ltot5.insert(k,tag)
            k=k+1	
    sent_tags=Ltot5
    return sen_words, sent_tags

def get_tags2(example):
    example.TokenList2=ast.literal_eval(example.TokenList2)
    sen_words=example.TokenList2
    L2=str(example.pattern2).split()
    Ltot5=[]
    k=0
    for i in range(0,len(sen_words)):
        tagf=L2[i]
        if isfloat(sen_words[i])==True:
           sen_words[i]=str(int(sen_words[i]))
        Lp=str(sen_words[i]).split()
        if len(Lp)>1:
            tag='B-'+tagf
            Ltot5.insert(k,tag)
            k=k+1
            for j in range(1,len(Lp)):
                tag='I-'+tagf
                Ltot5.insert(k,tag)
                k=k+1
        else:
            tag='B-'+tagf
            Ltot5.insert(k,tag)
            k=k+1
    sent_tags=Ltot5
    return sen_words, sent_tags