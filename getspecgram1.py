
import numpy as np
import wave
from melbank import *

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.14.0 


class GetSpecgram:
	"""  get spectrogram """
	def __init__(self,NFFT=1024, NSHIFT = 256,sampling_rate=22050, num_banks=64, f1=50, f2=5000):
		
		self.NFFT=NFFT
		self.NSHIFT=NSHIFT
		self.num_banks=num_banks
		self.sampling_rate=sampling_rate
		self.num_fft_bands=(NFFT/2)+1
		self.f1=f1
		self.f2=f2
		#
		self.melmat ,(melfreq, fftfreq)= compute_melmat(self.num_banks, self.f1, self.f2, self.num_fft_bands,self.sampling_rate)
		#
		self.window = np.hamming(self.NFFT)
		
		
	def get(self,file_name,fshow=False):
    	
		if fshow :
			print (file_name)
    	
		waveFile= wave.open( file_name, 'r')
		
		nchannles= waveFile.getnchannels()
		samplewidth = waveFile.getsampwidth()
		sampling_rate = waveFile.getframerate()
		nframes = waveFile.getnframes()
		
		assert sampling_rate == self.sampling_rate, ' sampling rate is miss match ! '
		
		if fshow :
			print("Channel num : ", nchannles)
			print("Sampling rate : ", sampling_rate)
			print("Frame num : ", nframes)
			print("Sample width : ", samplewidth)
		
		buf = waveFile.readframes(-1) # read all, or readframes( 1024)
		
		waveFile.close()
		
		if samplewidth == 2:
			data = np.frombuffer(buf, dtype='int16')
			fdata = data.astype(np.float32) / 32768.
		elif samplewidth == 4:
			data = np.frombuffer(buf, dtype='int32')
			fdata = data.astype(np.float32) / np.power(2.0, 31)
		
		# To MONO
		if nchannles == 2:
			#l_channel = fdata[::nchannles]
			#r_channel = fdata[1::nchannles]
			fdata= (fdata[::nchannles] + fdata[1::nchannles]) /2.0
		
		count= ((nframes - (self.NFFT - self.NSHIFT)) / self.NSHIFT)
		time_song = float(nframes) / sampling_rate
		time_unit = 1 / float(sampling_rate)
		
		if fshow :
			print("time song : ", time_song)
			print("time unit : ", time_unit)
			print("count : ", count)
		
		# initi spect 
		spec = np.zeros([count,self.num_banks]) 
		pos = 0
		countr=0
		for fft_index in range(count):
			frame = fdata[pos:pos+self.NFFT]
			
			if len(frame) == self.NFFT:
				windowed = self.window * frame
				fft_result = np.fft.fft(windowed)
				fft_data = np.abs(fft_result)
				fft_data2 = np.dot(fft_data[0:self.NFFT/2+1], self.melmat.T)
				
				if np.count_nonzero(fft_data2) == len(fft_data2):
					fft_data2 = np.log10(fft_data2) * 20  # dB
					
					for i in range(self.num_banks):
						spec[countr][i] = fft_data2[i]
					# index count up
					countr +=1
				else:
					print (' there is zero data. skiiped. ')
					
				# next
				pos += self.NSHIFT
		
		
		#spec = spec.reshape([countr,self.num_banks])
		#print (spec.shape, 'countr ' ,countr)
		# real data only
		spec = spec[0:countr,:]
		
		if fshow:
			print ("max,min", np.max(spec), np.min(spec) )

		return spec


# this file use TAB

