import matplotlib.pyplot as plt



log_path = "../logs/"
#filename = "log.traintest_03-18_17-08-23.3DCE 1 image 3 slice" #previous run, random 30%, no annealing, 10 epochs

#filename = "log.traintest_04-16_08-08-30.3DCE 1 image 3 slice" #Smaller test run, 10 epochs 10% sampling
filename = "log.traintest_04-16_09-45-43.3DCE 1 image 3 slice" #Latest run with sim_annealing, 10% sampling


epoch=[]
epoch_time=[]
epoch_time_accumulated=[]
epoch_validate=[]
epochn=0

fio = open(log_path+filename, "r")
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
    if 'Sensitivity' in line:
        #print(line)
        s=line.split(':')
        s[-1] = s[-1].strip() #remove the \n
        r = s[-1].split()
        #print(r)
        epoch_validate.append(float(r[3]))


fio.close()

print(epoch)
print(epoch_time)
print(epoch_time_accumulated)
print(epoch_validate)

plt.plot(epoch, epoch_validate)
plt.ylabel('sensitivity')
plt.xlabel('# of epochs')
plt.show()


plt.plot(epoch_time_accumulated, epoch_validate)
plt.ylabel('sensitivity')
plt.xlabel('epoch_time (s)')
plt.show()