import matplotlib.pyplot as plt



log_path = "../logs/"
#filename = "log.traintest_03-18_17-08-23.3DCE 1 image 3 slice" #previous run, random 30%, no annealing, 10 epochs

#filename = "log.traintest_04-16_08-08-30.3DCE 1 image 3 slice" #Smaller test run, 10 epochs 10% sampling
filename = "log.traintest_04-16_09-45-43.3DCE 1 image 3 slice" #Latest run with sim_annealing, 10% sampling


epoch=[]
epoch_time=[]
epoch_time_accumulated=[]
epoch_sensitivity=[] # Same as recall
epoch_precision=[]
epoch_recall=[] # same as racall
epoch_fmeasure=[]
epochn=0

fio = open(log_path+filename, "r")
prev_line=""
for line in fio:
    if 'Time cost' in line:
        #print(line)
        s = line.split('=')
        s[-1] = s[-1].strip() #remove the \n
        #print(s)
        epoch_time.append(float(s[1]))
        epoch.append(float(epochn))
        epochn=epochn+1

        if len(epoch_time_accumulated) == 0:
            epoch_time_accumulated.append(float(s[1]))
        else:
            epoch_time_accumulated.append(epoch_time_accumulated[-1]+float(s[1]))
        
    #Note: The validation criteria for 3DCE uses Sensitivity @ 4, in the log files.
    #Note2: Sensitivity is the same as Recall    
    if 'Sensitivity' in line:
        #print(line)
        s=line.split(':')
        s[-1] = s[-1].strip() #remove the \n
        r = s[-1].split()
        #print(r)
        epoch_sensitivity.append(float(r[3]))

    # The log file prints the results below the string, so query prev line
    if 'precision' in prev_line:
        s=line.split()
        epoch_precision.append(float(s[3]))
    if 'recall' in prev_line:
        s=line.split()
        epoch_recall.append(float(s[3]))
    if 'f-measure' in prev_line:
        s=line.split()
        epoch_fmeasure.append(float(s[3]))

    prev_line=line

fio.close()

#print(epoch)
#print(epoch_time)
#print(epoch_time_accumulated)
#print(epoch_sensitivity)
#print(epoch_precision)
#print(epoch_recall)
#print(epoch_fmeasure)

#plt.plot(epoch, epoch_sensitivity, epoch, epoch_precision, epoch, epoch_fmeasure)
plt.plot(epoch, epoch_sensitivity )
plt.ylabel('recall (sensitivity)')
plt.xlabel('# of epochs')
plt.show()


plt.plot(epoch_time_accumulated, epoch_sensitivity)
plt.ylabel('recall (sensitivity)')
plt.xlabel('epoch_time (s)')
plt.show()