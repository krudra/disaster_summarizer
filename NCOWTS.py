import sys
from collections import Counter
import re
from gurobipy import *
import gzip
import networkx as nx
from textblob import *
import os
import time
import codecs
import math
import codecs
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic, genesis
import numpy as np
import aspell
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import pylab as pl
from itertools import cycle
from operator import itemgetter


LOWLIMIT = 1
UPLIMIT = 100
LSIM = 0.7
SIM_TH = 0.7
Tagger_Path = '/home/krudra/summarization/summary_ilp/ark-tweet-nlp-0.3.2/'
ASPELL = aspell.Speller('lang', 'en')
WORD = re.compile(r'\w+')
cachedstopwords = stopwords.words("english")
INFOPATH = '/home/krudra/summarization/sigir/infomap/'
PATH = '/home/krudra/summarization/sigir/Ijcai/data/build_weight/'

def compute_similarity(ifname,breakfile,placefile,keyterm):

	B = {}
	fp = open(breakfile,'r')
	for l in fp:
		wl = l.split()
		#print(wl)
		B[int(wl[1].strip(' \t\n\r'))] = int(wl[2])
	fp.close()

	PLACE = {}
        fp = codecs.open(placefile,'r','utf-8')
        for l in fp:
                if PLACE.__contains__(l.strip(' #\t\n\r').lower())==False:
                	PLACE[l.strip(' #\t\n\r').lower()] = 1
        fp.close()

	fp = codecs.open(ifname,'r','utf-8')
	T = {}
	TW = {}
	M = {}
	P = []
	index = 0
	count = 0
	word = {}
	coword = {}
	window = 0
	t0 = time.time()
	
	for l in fp:
		count+=1
		wl = l.split('\t')
		temp_I = wl[5].split()
		All_I = wl[6].split()
		text = wl[4].strip(' \t\n\r')
		Length = int(wl[7])
		tid = wl[2].strip(' \t\n\r')
		temp = []
		for x in temp_I:
			x_0 = x.split('_')[0].strip(' \t\n\r')
			x_1 = x.split('_')[1].strip(' \t\n\r')
			if x_1=='PN':
                                s = x_0 + '_CN'
                                temp.append(s)
                        else:
                                temp.append(x)

			#temp.append(x)
			'''if x_1=='$':
                                try:
                                        p_1 = int(x_0)
                                        temp.append(x)
                                except Exception as e:
                                        pass
                        else:
                                #if REJECT.__contains__(x_0)==False:
                                temp.append(x)'''
			'''if x_1=='P':
				if PLACE.__contains__(x_0)==True:
					temp.append(x)
			else:
				temp.append(x)'''
		All = []
		for x in All_I:
			x_0 = x.split('_')[0].strip(' \t\n\r')
			x_1 = x.split('_')[1].strip(' \t\n\r')
			if x_1=='PN':
                                s = x_0 + '_CN'
                                All.append(s)
                        else:
                                All.append(x)
			#All.append(x)
			'''if x_1=='$':
                                try:
                                        p_1 = int(x_0)
                                        All.append(x)
                                except Exception as e:
                                        pass
                        else:
                                #if REJECT.__contains__(x_0)==False:
                                All.append(x)'''
			'''if x_1=='P':
				if PLACE.__contains__(x_0)==True:
					All.append(x)
			else:
				All.append(x)'''
		#k = should_select(T,All)

		################### Update word dictionary  ###################################
		for x in temp:
			if word.__contains__(x)==True:
				v = word[x]
				v+=1
				word[x] = v
			else:
				word[x] = 1
		
		################## Is it duplicate tweet ######################################

		#k = compute_cosine_similarity(All,TW)
		k = compute_selection_criteria(All,TW)
		if k==1:
			T[index] = temp
			TW[index] = [tid,text,temp,All,Length]
			index+=1
		
		#if B.__contains__(tid)==True:
		if B.__contains__(count)==True:
			print('Breakpoint: ',count, index)
			L = len(T.keys())
			print('L: ',L)
			weight = compute_tfidf(word,count,PLACE)
			tweet_cur_window = {}
			check = set([])
                        for i in range(0,L,1):
                                temp = TW[i]
                                #tweet_cur_window[temp[0].strip(' \t\n\r')] = [temp[1],temp[4],set(temp[2])]   ### Text, Length, Content words ###
                                tweet_cur_window[str(i)] = [temp[1],temp[4],set(temp[2])]   ### Text, Length, Content words ###
				for x in temp[2]:
					check.add(x)
			print('Content: ',len(check))
                        ##################### Finally apply cowts ################################
                        ofname = keyterm + '_COWTS_' + str(window) + '.txt'
                        optimize(tweet_cur_window,weight,ofname,B[count],0.4,0.6)
                        print('Summarization done: ',ofname)
                        window+=1
                        t1 = time.time()
                        print('Time Elapsed: ',t1-t0)
	fp.close()

def compute_selection_criteria(current,T):
        for k,v in T.iteritems():
                new = set(current).intersection(set(v[3]))
                if len(new)==len(current):
                        return 0
        return 1

def wordnet_similarity(word1,word2,brown_ic,semcor_ic,genesis_ic):
	
	'''sim_1 = word1.lin_similarity(word2, brown_ic)
	sim_2 = word1.lin_similarity(word2, semcor_ic)
	sim_3 = word1.lin_similarity(word2, genesis_ic)
	sim = max(sim_1,sim_2,sim_3)
	return sim'''

	cbv = wordnet.synsets(word1)
	ibv = wordnet.synsets(word2)
        #print(cbv, ibv)

        MAX = []
        for i in range(0,len(cbv),1):
                w1 = wordnet.synset(cbv[i].name())
        	for j in range(0,len(ibv),1):
                        w2 = wordnet.synset(ibv[j].name())
                        #MAX.append(w1.wup_similarity(w2))
			try:
				sim_1 = w1.lin_similarity(w2, brown_ic)
				#sim_1 = w1.lin_similarity(w2, semcor_ic)
				#sim_2 = w1.lin_similarity(w2, semcor_ic)
				#sim_3 = w1.lin_similarity(w2, genesis_ic)
				#MAX.append(max(sim_1,sim_2,sim_3))
				MAX.append(sim_1)
			except Exception as e:
				pass
			
	#print(max(MAX))
	try:
		m = max(MAX)
		if m==None:
			m = 0
		return m
	except Exception as e:
		return 0

def topic_pageweight(nodeweight,N,coweight,NOUN,VERB):

        G = nx.Graph()
        #print('No of Tweets: ',len(T))
        NW = {}
        for i in range(0,len(N),1):
                x = N[i]
                if G.has_node(x)==False and nodeweight.__contains__(x)==True:
                        G.add_node(x,score=nodeweight[x])
                        NW[x] = nodeweight[x]

        #print(G.nodes())
        #print(Nodes)
	TAG = ['S','P']
        for i in range(0,len(N)-1,1):
		x_1 = N[i].split('_')[1].strip(' \t\n\r')
                for j in range(i+1,len(N),1):
			x_2 = N[j].split('_')[1].strip(' \t\n\r')
			if x_1 in TAG and x_2 in TAG:
                        	if G.has_edge(N[i],N[j])==False:
                                	if coweight.__contains__((N[i],N[j]))==True:
                                        	G.add_edge(N[i],N[j],weight=coweight[(N[i],N[j])])
                                	elif coweight.__contains__((N[j],N[i]))==True:
                                        	G.add_edge(N[i],N[j],weight=coweight[(N[j],N[i])])
                                	else:
						pass
                                        	#G.add_edge(N[i],N[j],weight=0.001)
			elif x_1 in TAG and x_2 not in TAG:
				m = 0
				TYPE = N[j].split('_')[0].strip(' \t\n\r')
				if TYPE=='topic':
					temp = NOUN[int(x_2)]
					for x in temp:
						if coweight.__contains__((N[i],x))==True:
							m += coweight[(N[i],x)]
						elif coweight.__contains__((x,N[i]))==True:
							m += coweight[(x,N[i])]
						else:
							pass
					p1 = m + 4.0 - 4.0
					p2 = len(temp) + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				elif TYPE=='event':
					temp = VERB[int(x_2)]
					for x in temp:
						if coweight.__contains__((N[i],x))==True:
							m += coweight[(N[i],x)]
						elif coweight.__contains__((x,N[i]))==True:
							m += coweight[(x,N[i])]
						else:
							pass
					p1 = m + 4.0 - 4.0
					p2 = len(temp) + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				else:
					pass
			elif x_1 not in TAG and x_2 in TAG:
				m = 0
				TYPE = N[i].split('_')[0].strip(' \t\n\r')
				if TYPE=='topic':
					temp = NOUN[int(x_1)]
					for x in temp:
						if coweight.__contains__((N[j],x))==True:
							m += coweight[(N[j],x)]
						elif coweight.__contains__((x,N[j]))==True:
							m += coweight[(x,N[j])]
						else:
							pass
					p1 = m + 4.0 - 4.0
					p2 = len(temp) + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				elif TYPE=='event':
					temp = VERB[int(x_1)]
					for x in temp:
						if coweight.__contains__((N[j],x))==True:
							m += coweight[(N[j],x)]
						elif coweight.__contains__((x,N[j]))==True:
							m += coweight[(x,N[j])]
						else:
							pass
					p1 = m + 4.0 - 4.0
					p2 = len(temp) + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				else:
					pass
			else:
				TYPE_1 = N[i].split('_')[0].strip(' \t\n\r')
				TYPE_2 = N[j].split('_')[0].strip(' \t\n\r')
				if TYPE_1=='topic' and TYPE_2=='topic':
					temp_1 = NOUN[int(x_1)]
					temp_2 = NOUN[int(x_2)]
					m = 0
					p2 = 0
					for x in temp_1:
						for y in temp_2:
							p2+=1
							if coweight.__contains__((x,y))==True:
								m += coweight[(x,y)]
							elif coweight.__contains__((y,x))==True:
								m += coweight[(y,x)]
							else:
								pass
					p1 = m + 4.0 - 4.0
					p2 = p2 + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				elif TYPE_1=='event' and TYPE_2=='event':
					temp_1 = VERB[int(x_1)]
					temp_2 = VERB[int(x_2)]
					m = 0
					p2 = 0
					for x in temp_1:
						for y in temp_2:
							p2+=1
							if coweight.__contains__((x,y))==True:
								m += coweight[(x,y)]
							elif coweight.__contains__((y,x))==True:
								m += coweight[(y,x)]
							else:
								pass
					p1 = m + 4.0 - 4.0
					p2 = p2 + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				elif TYPE_1=='topic' and TYPE_2=='event':
					temp_1 = NOUN[int(x_1)]
					temp_2 = VERB[int(x_2)]
					m = 0
					p2 = 0
					for x in temp_1:
						for y in temp_2:
							p2+=1
							if coweight.__contains__((x,y))==True:
								m += coweight[(x,y)]
							elif coweight.__contains__((y,x))==True:
								m += coweight[(y,x)]
							else:
								pass
					p1 = m + 4.0 - 4.0
					p2 = p2 + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				else:
					temp_1 = VERB[int(x_1)]
					temp_2 = NOUN[int(x_2)]
					m = 0
					p2 = 0
					for x in temp_1:
						for y in temp_2:
							p2+=1
							if coweight.__contains__((x,y))==True:
								m += coweight[(x,y)]
							elif coweight.__contains__((y,x))==True:
								m += coweight[(y,x)]
							else:
								pass
					p1 = m + 4.0 - 4.0
					p2 = p2 + 4.0 - 4.0
					avg = round(p1/p2,4)
					if G.has_edge(N[i],N[j])==False:
						G.add_edge(N[i],N[j],weight=avg)
				
	#print(G.edges(data=True))
        pr = nx.pagerank(G,max_iter=200,nstart=NW,weight='weight')
	#print(type(pr))
	#sys.exit(0)
	return pr
        #pr = nx.pagerank(G,nstart=nodeweight)

def affinity_clustering(S):
        af = AffinityPropagation().fit(S)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(cluster_centers_indices)
        #print 'Estimated number of clusters: %d' % n_clusters_

        '''print 'Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, labels)
        print 'Completeness: %0.3f' % metrics.completeness_score(labels_true, labels)
        print 'V-measure: %0.3f' % metrics.v_measure_score(labels_true, labels)
        print 'Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, labels)
        print 'Adjusted Mutual Information: %0.3f' % metrics.adjusted_mutual_info_score(labels_true, labels)
        D = (S / np.min(S))
        print ('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(D, labels, metric='precomputed'))'''
        #print(cluster_centers_indices)
        #print(labels)

        L = [cluster_centers_indices,labels]

        #return cluster_centers_indices
        return L

        '''for k, col in zip(range(n_clusters_), colors):
                class_members = labels == k
                cluster_center = X[cluster_centers_indices[k]]
                pl.plot(X[class_members, 0], X[class_members, 1], col + '.')
                pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
                for x in X[class_members]:
                        pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

        pl.title('Estimated number of clusters: %d' % n_clusters_)
        pl.show()'''


def compute_coweight(coword,word,count):
	coscore = {}
	exc = 0
	for k,v in coword.iteritems():
		s1 = k[0]
		s2 = k[1]
		if s1=='speak_V' and s2=='inform_V':
			print(s1,word[s1],s2,word[s2],v)
		#den = word[s1] * word[s2]
		#num = count * v
		den = word[s1] + word[s2] - v
		num = v
		x1 = num + 4.0 - 4.0
		x2 = den + 4.0 - 4.0
		try:
			coscore[k] = round(x1/x2,4)
		except Exception as e:
			coscore[k] = 0
			exc+=1
	print('Got exception ', exc, ' times')
	
	'''count = 0
	for k,v in coscore.iteritems():
		print(k,v)
		count+=1
		if count>=20:
			break
	sys.exit(0)'''
	return coscore

def compute_raw_tf(word,tweet_count,PLACE):
        score = {}
        discard = ['pray','prayer','updates','pls','please','hope','hoping','breaking','news','flash','update','tweet','pm','a/c','v/','w/o','watch','photo','video','picture','screen','pics','latest','plz','rt','mt','follow','tv','pic','-mag','cc','-please','soul','hoax','a/n','utc','some','something','ist','afr','guru','image','images']

        #THR = 0
        #N = tweet_count + 4.0 - 4.0
        for k,v in word.iteritems():
		D = k.split('_')
		D_w = D[0].strip(' \t\n\r')
		D_t = D[1].strip(' \t\n\r')
                if D_w not in discard:
			score[k] = v
		else:
			score[k] = 0
        return score

def compute_tf(word):
        score = {}
        discard = ['pray','prayer','updates','pls','please','hope','hoping','breaking','news','flash','update','tweet','pm','a/c','v/','w/o','watch','photo','video','picture','screen','pics','latest','plz','rt','mt','follow','tv','pic','-mag','cc','-please','soul','hoax','a/n','utc','some','something','ist','afr','guru','image','images']

        #THR = 0
        #N = tweet_count + 4.0 - 4.0
        for k,v in word.iteritems():
		D = k.split('_')
		D_w = D[0].strip(' \t\n\r')
		D_t = D[1].strip(' \t\n\r')
                if D_w not in discard:
                        tf = v
                        w = 1 + math.log(tf,2)
			score[k] = round(w,4)
		else:
			score[k] = 0
        return score

def rank_nodes(simfile,weight):

        sim = {}
        keys = set([])
        fp = open(simfile,'r')
        for l in fp:
                wl = l.split()
                t = (wl[0].strip(' \t\n\r'),wl[1].strip(' \t\n\r'))
                score = float(wl[2])
                if sim.__contains__(t)==False:
                        sim[t] = score
                keys.add(wl[0].strip(' \t\n\r'))
                keys.add(wl[1].strip(' \t\n\r'))
        fp.close()

        threshold = 0
        tolerance = 0.00001

        content = list(weight.keys())
        adjacency_matrix = np.zeros([len(content), len(content)])
        for i in range(0,len(content)-1,1):
                temp = []
                degree = 0.0
                for j in range(0,len(content),1):
                        t = (content[i],content[j])
                        t1 = (content[j],content[i])
                        if sim.__contains__(t)==True and sim[t]>0:
                                temp.append(sim[t])
                                degree+=sim[t]
                        elif sim.__contains__(t1)==True and sim[t1]>0:
                                temp.append(sim[t1])
                                degree+=sim[t1]
                        elif content[i]==content[j]:
                                temp.append(1.0)
                                degree+=1.0
                        else:
                                temp.append(0)
                #norm = []
                den = degree + 4.0 - 4.0
                for k in range(0,len(temp),1):
                        x = temp[k] + 4.0 - 4.0
                        z = round(x/den,4)
                        adjacency_matrix[i][k] = z
                        #norm.append(z)
                #adjacency_matrix.append(norm)
	print(adjacency_matrix[0])

        temp_scores = np.zeros([len(content)])
	count = 0
	for i in range(0,len(content),1):
		temp_scores[i] = weight[content[i]]
		count+=weight[content[i]]
	scores = np.zeros([len(content)])
	for i in range(0,len(content),1):
		x = round(temp_scores[i]/count,4)
		scores[i] = x
        page_score = power_method(adjacency_matrix, scores, tolerance)

	dic_score = {}
	for i in range(0,len(content),1):
		dic_score[content[i]] = page_score[i]

	count = 0
	for i in range(0,10,1):
		print(content[i],scores[i],page_score[i])
	'''for k,v in dic_score.iteritems():
		print(k,v)
		count+=1
		if count==5:
			break'''
	return dic_score

def power_method(m, scores, epsilon):
        n = len( m )
        #p = [1.0 / n] * n
        p = scores
        Iter = 0
        while True:
                new_p = [0] * n
                for i in xrange( n ):
                        for j in xrange( n ):
                                new_p[i] += m[j][i] * p[j]
                total = 0
                for x in xrange( n ):
                        total += ( new_p[i] - p[i] ) ** 2
                p = new_p
                Iter+=1
                print('Iteration Number: ',Iter)
                if total < epsilon:
                         break
                if Iter>=500:
                        break
        return p

'''
def pageweight(nodeweight,T,coweight):
	
	G = nx.Graph()
	print('No of Tweets: ',len(T))
	NW = {}
	for i in range(0,len(T),1):
		temp = T[i]
		for x in temp:
			if G.has_node(x)==False and nodeweight.__contains__(x)==True:
				G.add_node(x,score=nodeweight[x])
				NW[x] = nodeweight[x]
	Nodes = G.nodes()
	#print(Nodes)
	for i in range(0,len(Nodes)-1,1):
		for j in range(i+1,len(Nodes),1):
			if G.has_edge(Nodes[i],Nodes[j])==False and nodeweight.__contains__(Nodes[i])==True and nodeweight.__contains__(Nodes[j])==True:
				if coweight.__contains__((Nodes[i],Nodes[j]))==True:
					G.add_edge(Nodes[i],Nodes[j],weight=coweight[(Nodes[i],Nodes[j])])
				else:
					G.add_edge(Nodes[i],Nodes[j],weight=0.001)
			else:
				G.add_edge(Nodes[i],Nodes[j],weight=0.001)
	pr = nx.pagerank(G,max_iter=200,nstart=NW,weight='weight')
	#pr = nx.pagerank(G,nstart=nodeweight)
	return pr
'''

def set_weight(weight,L,U):
	temp = []
	for k,v in weight.iteritems():
		temp.append(v)
	temp.sort()
	min_p = temp[0]
	max_p = temp[len(temp)-1]
	
	x = U - L + 4.0 - 4.0
	y = max_p - min_p + 4.0 - 4.0
	factor = round(x/y,4)
	
	mod_weight = {}
	for k,v in weight.iteritems():
		p = L + factor * (v - min_p)
		mod_weight[k] = round(p,4)
	
	'''count = 0	
	for k,v in mod_weight.iteritems():
		print(k,v)
		count+=1
		if count==10:
			break'''
	return mod_weight

def compute_norm(word):
	
	new_word = {}
	for k,v in word.iteritems():
		#p = math.log10(v)
		p = v
		if p==0:
			p = 1
		new_word[k] = v

	temp = []
	for k,v in new_word.iteritems():
		temp.append(v)
	M =  max(temp)
	y = M + 4.0 - 4.0
	F = {}
	for k,v in new_word.iteritems():
		x = v + 4.0 - 4.0
		z = round(x/y,4)
		F[k] = z
	count = 0
	for k,v in F.iteritems():
		print(k,v)
		count+=1
		if count==5:
			break
	return F

def should_select(new,M):
	flag = 0
	for x in new:
		if M.__contains__(x)==False:
			flag+=1
	if flag>0:
		return 1
	return 0
		
def build_community(T):
	L = len(T.keys())
	sim = {}
	for i in range(0,L,1):
		F1 = T[i]
		for j in range(i,L,1):
			F2 = T[j]
			s = select_cluster(F1,F2)
			sim[(i,j)] = round(s,4)
	S = []
	for i in range(0,L,1):
		temp = []
		for j in range(0,L,1):
			if i<=j:
				temp.append(sim[(i,j)])
			else:
				temp.append(sim[(j,i)])
		S.append(temp)
	
	index = 0
	fo = open('temp_input.txt','w')
	for List in S:
		for i in range(0,len(List),1):
			if index!=i and i > index:
				if List[i] > 0:
					s = str(index) + ' ' + str(i) + ' ' + str(List[i])
					fo.write(s)
					fo.write('\n')
		index+=1
	fo.close()

	command = INFOPATH + './Infomap temp_input.txt out/ -N 10  --zero-based-numbering --two-level --clu'
	os.system(command)

	fp = open('out/temp_input.clu','r')
	index = 0
	community = {}
	for l in fp:
		if index>=2:
			wl = l.split()
			c = int(wl[1])
			if community.__contains__(c)==True:
				v = community[c]
				v.append(int(wl[0]))
				community[c] = v
			else:
				community[c] = [int(wl[0])]
		index+=1
	'''command = 'rm temp_input.txt'
	os.system(command)'''
	return community
				
def optimize(tweet,weight,ofname,L,A1,A2):


	################################ Extract Tweets and Content Words ##############################
	word = {}
	tweet_word = {}
	tweet_index = 1
	for  k,v in tweet.iteritems():
		set_of_words = v[2]
		for x in set_of_words:
			if word.__contains__(x)==False:
				if weight.__contains__(x)==True:
					p1 = round(weight[x],4)
				else:
					p1 = 0.0
				word[x] = p1

		tweet_word[tweet_index] = [v[1],set_of_words,v[0]]  #Length of tweet, set of content words present in the tweet, tweet itself
		tweet_index+=1

	############################### Make a List of Tweets ###########################################
	sen = tweet_word.keys()
	sen.sort()
	entities = word.keys()
	print(len(sen),len(entities))

	################### Define the Model #############################################################

	m = Model("sol1")

	############ First Add tweet variables ############################################################

	sen_var = []
	for i in range(0,len(sen),1):
		sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))
		'''var = 'x' + str(sen[i])
		var1 = '"' + var + '"'
		var = m.addVar(vtype=GRB.BINARY, name=var1)'''

	############ Add entities variables ################################################################

	con_var = []
	for i in range(0,len(entities),1):
		con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))
		'''var = 'y' + str(i+1)
		var1 = '"' + var + '"'
		var = m.addVar(vtype=GRB.BINARY, name=var1)'''

	########### Integrate Variables ####################################################################
	m.update()

	'''for i in range(0,len(entities),1):
		print(i,entities[i])
	sys.exit(0)'''

	P = LinExpr() # Contains objective function
	C1 = LinExpr()  # Summary Length constraint
	C4 = LinExpr()  # Summary Length constraint
	C2 = [] # If a tweet is selected then the content words are also selected
	counter = -1
	for i in range(0,len(sen),1):
		#P += tweet_word[i+1][0] * sen_var[i]
		P += sen_var[i]
		C1 += tweet_word[i+1][0] * sen_var[i]
		#C1 += sen_var[i]
		v = tweet_word[i+1][1] # Entities present in tweet i+1
		#print(v)
		C = LinExpr()
		flag = 0
		for j in range(0,len(entities),1):
			if entities[j] in v:
				flag+=1
				C += con_var[j]
		if flag>0:
			#print(C,flag)
			counter+=1
			m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))
		#s = C + GRB.GREATER_EQUAL + str(len(v))
		#C2.append(s)
		#decl = decl + 'x' + str(i+1) + ' , '

	C3 = [] # If a content word is selected then at least one tweet is selected which contains this word
	#C4 = ''
	for i in range(0,len(entities),1):
		P += word[entities[i]] * con_var[i]
		#C4 = C4 + 'y' + str(i+1) + ' + '
		C = LinExpr()
		flag = 0
		for j in range(0,len(sen),1):
			v = tweet_word[j+1][1]
			if entities[i] in v:
				flag = 1
				C += sen_var[j]
		if flag==1:
			counter+=1
			m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))
		#s = LinExpr()
		#s = C + ' >= ' + con_var[i]
		#C3.append(s)
		#decl = decl + 'y' + str(i+1) + ' , '

	#s = P.strip(' +') + ' );'
	#s = P.strip(' +')
	#P = s
	#fo.write(P)
	#fo.write('\n')
	#m.addConstr(C1,GRB.LESS_EQUAL,L,"c0")
	#K = 40
	counter+=1
	m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))


	################ Set Objective Function #################################
	#print(P)
	m.setObjective(P, GRB.MAXIMIZE)
	#m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

	############### Set Constraints ##########################################

	fo = codecs.open(ofname,'w','utf-8')
	try:
		m.optimize()
		for v in m.getVars():
			if v.x==1:
				temp = v.varName.split('x')
				if len(temp)==2:
					fo.write(tweet_word[int(temp[1])][2])
					fo.write('\n')
	except GurobiError as e:
    		print(e)
		sys.exit(0)

	#fp.close()
	fo.close()

def compute_tfidf(word,tweet_count,PLACE):
        score = {}
        #discard = ['pray','prayer','updates','pls','please','hope','hoping','breaking','news','flash','update','tweet','pm','a/c','v/','w/o','watch','photo','video','picture','screen','pics','latest','plz','rt','mt','follow','tv','pic','-mag','cc','-please','soul','hoax','a/n','utc','some','something','ist','afr','guru','image','images']

	discard = []
        #THR = int(round(math.log10(tweet_count),0))
        THR = 5
        N = tweet_count + 4.0 - 4.0
        for k,v in word.iteritems():
		D = k.split('_')
		D_w = D[0].strip(' \t\n\r')
		D_t = D[1].strip(' \t\n\r')
                if D_w not in discard:
                        tf = v
                        w = 1 + math.log(tf,2)
                        df = v + 4.0 - 4.0
                        #N = tweet_count + 4.0 - 4.0
                        try:
                                y = round(N/df,4)
                                idf = math.log10(y)
                        except Exception as e:
                                idf = 0
                        val = round(w * idf, 4)
                        if D_t=='P' and tf>=THR:
                                score[k] = val
                        elif tf>=THR and D_t=='S':
                                score[k] = val
			elif tf>=THR and len(D_w)>2:
				score[k] = val
                        else:
                                score[k] = 0
                else:
                        score[k] = 0
        return score

def compute_cosine_similarity(F1,Total):
        s = ''
        for x in F1:
		wl = x.split('_')
		word = wl[0].strip(' \t\n\r')
		tag = wl[1].strip(' \t\n\r')
		s = s + word + ' '
		#if tag=='N' or tag=='V':
                #	s = s + x + ' '
        s1 = s.strip(' ')
        vec1 = text_to_vector_tf(s1) #Convert new one'''

	s = ''
	SIM = []
	for k,v in Total.iteritems():
		F2 = v[3]
		s = ''
		for x in F2:
			wl = x.split('_')
			word = wl[0].strip(' \t\n\r')
			tag = wl[1].strip(' \t\n\r')
			s = s + word + ' '
		s1 = s.strip(' ')
		vec2 = text_to_vector_tf(s1)
		cos = get_cosine(vec1,vec2)
		SIM.append(cos)
	
	M = 0
	if len(SIM)==0:
		M = 0
	else:
		M = max(SIM)

	if M < SIM_TH:
		return 1
	return 0

def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
                return 0.0
        else:
                return float(numerator) / denominator

def text_to_vector_tf(text):
        words = WORD.findall(text)
        return Counter(words)

def get_max_sim(T,index):
	
	if index==0:
		return 0

	temp = T[index][4]
	s = ''
	for x in temp:
		s = s + x + ' '
	s1 = s.strip(' ')
        vec1 = text_to_vector_tf(s1) #Convert new one'''
	
	SIM = []
	for i in range(0,index,1):
		s = ''
		temp = T[i][4]
		for x in temp:
			s = s + x + ' '
		s1 = s.strip(' ')
        	vec2 = text_to_vector_tf(s1)
		cos = get_cosine(vec1,vec2)
		SIM.append(cos)
	SIM.sort()
	SIM.reverse()
	return SIM[0]

def apsalience(tweet,ofname,L):
	T = []
	s = ''
	for k,v in tweet.iteritems():
		T.append((k,v[0],v[1],v[2]))

	T.sort(key=itemgetter(3),reverse=True)

	fo = codecs.open('getR.txt','w','utf-8')
	for x in T:
		fo.write(x[1])
		fo.write('\n')
	fo.close()

	command = Tagger_Path + './runTagger.sh --output format conll getR.txt > tag.txt'
	os.system(command)
	
	P = []
	TAGREJECT = ['@','#',',','~','U','E','G']
	fp = codecs.open('tag.txt','r','utf-8')
	index = 0
	temp = []
	for l in fp:
		wl = l.split()
		if len(wl) > 1:
			word = wl[0].strip(' #\t\n\r').lower()
			tag = wl[1].strip(' \t\n\r')
			if tag not in TAGREJECT and word not in cachedstopwords:
				temp.append(word)
		else:
			q = T[index]
			P.append((q[0],q[1],q[2],q[3],temp))
			index+=1
			temp = []
	fp.close()

	P.sort(key=itemgetter(3),reverse=True)
	count = 0
	fo = codecs.open(ofname,'w','utf-8')
	for i in range(0,len(P),1):
		sim = get_max_sim(P,i)
		if sim < LSIM:
			fo.write(P[i][1])
			fo.write('\n')
			count+=P[i][2]
			if count>=L:
				break
	fo.close()
		
def Normalize(M):
	m = max(M)
	y = m + 4.0 - 4.0
	P = []
	for i in range(0,len(M),1):
		x = M[i] + 4.0 - 4.0
		z = round(x/y,4)
		P.append(z)
	return P	

def compute_tfidf_not(word,tweet_count,PLACE):

        score = {}
        discard = ['pray','prayer','updates','pls','please','hope','hoping','breaking','news','flash','update','tweet','pm','a/c','v/','w/o','watch','photo','video','picture','screen','pics','latest','plz','rt','mt','follow','tv','pic','-mag','cc','-please','soul','hoax','a/n','utc','some','something','ist','afr','guru','image','images']

        #THR = int(round(math.log10(tweet_count),0))
        THR = 5
        for k,v in word.iteritems():
                if k not in discard:
                        tf = v[0]
                        w = 1 + math.log(tf,2)
                        df = len(v[1]) + 4.0 - 4.0
                        N = tweet_count + 4.0 - 4.0
                        try:
                                y = round(N/df,4)
                                idf = math.log10(y)
                        except Exception as e:
                                idf = 0
                        val = round(w * idf, 4)
                        #if tf>=5:
                                # Consider a word if it occurs more than 10 times or its tf-idf score crosses 10
                         #       score[k] = val
                        #else:
                        PX = re.findall(r'[\d]+',k)
                        if PLACE.__contains__(k)==True:
                                score[k] = val
                        elif len(PX)>=1 and k.find('/')==-1:
                                score[k] = val
                        elif ASPELL.check(k)==1 and tf>=THR:
                                score[k] = val
                        else:
                                score[k] = 0
                else:
                        score[k] = 0
        return score

def main():
	try:
		_, ifname, breakfile, placefile, keyterm = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	compute_similarity(ifname,breakfile,placefile,keyterm)
	print('Koustav Done')

if __name__=='__main__':
	main()
