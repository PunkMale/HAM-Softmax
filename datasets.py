import numpy
import os
import glob
import wave
import random
import soundfile
import torch
import torch.nn.functional as F
from scipy import signal
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tools import *


class BaseDataset(Dataset):
    def __init__(self, train_list=None, train_path=None, num_frames=None, augment=False, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames

        self.data_list = []
        self.data_label = []

        self.augment = augment
        if self.augment:
            self.musan_path = "/home/database/noise/musan"
            self.rir_path = "/home/database/noise/RIRS_NOISES/simulated_rirs"
            self.noisetypes = ['noise','speech','music']
            self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
            self.noiselist = {}
            
            augment_files   = glob.glob(os.path.join(self.musan_path,'*/*/*.wav'))
            for file in augment_files:
                if file.split('/')[-3] not in self.noiselist:
                    self.noiselist[file.split('/')[-3]] = []
                self.noiselist[file.split('/')[-3]].append(file)
            self.rir_files  = glob.glob(os.path.join(self.rir_path,'*/*/*.wav'))
            for key in self.noiselist.keys():
                print(f"{key}: {len(self.noiselist[key])}")

        if train_list is None:
            print('create a empty dataset')
        else:
            lines = open(train_list).read().splitlines()
            dictkeys = list(set([x.split()[0] for x in lines]))
            dictkeys.sort()
            dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
            for index, line in enumerate(lines):
                speaker_id = line.split()[0]
                speaker_label = dictkeys[speaker_id]
                file_name = os.path.join(train_path, line.split()[1])
                self.data_label.append(speaker_label)
                self.data_list.append(file_name)
            print('speaker number:{}'.format(self.get_speaker_number()))
            print('utterance number:{}'.format(len(self.data_label)))
        self.data_label = torch.tensor(self.data_label, dtype=torch.long)

    def __getitem__(self, index):
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)

        if self.augment:
            augtype = random.randint(1,4)
            if augtype == 1:
                aug_audio = self.add_rev(audio, length=length)
            elif augtype == 2:
                aug_audio = self.add_noise(audio, 'music', length=length)
            elif augtype == 3:
                aug_audio = self.add_noise(audio, 'speech', length=length)
            elif augtype == 4:
                aug_audio = self.add_noise(audio, 'noise', length=length)
            return index, torch.FloatTensor(audio[0]), torch.FloatTensor(aug_audio[0]), self.data_label[index]
        else:
            return index, torch.FloatTensor(audio[0]), self.data_label[index]
    
    def get_augment_data(self, audio, length):
        augtype = random.randint(1,4)
        if augtype == 1:
            aug_audio = self.add_rev(audio, length=length)
        elif augtype == 2:
            aug_audio = self.add_noise(audio, 'music', length=length)
        elif augtype == 3:
            aug_audio = self.add_noise(audio, 'speech', length=length)
        elif augtype == 4:
            aug_audio = self.add_noise(audio, 'noise', length=length)
        else:
            raise ValueError('augment type error')
        return aug_audio
    
    def add_rev(self, audio, length):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float32),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:length]

    def add_noise(self, audio, noisecat, length):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, 'rb').getnframes()
            if noiselength <= length:
                noiseaudio, _ = soundfile.read(noise)
                noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
            else:
                start_frame = numpy.int64(random.random()*(noiselength-length))
                noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio

    def __len__(self):
        return len(self.data_list)

    def get_speaker_number(self):
        return max(self.data_label) + 1
    
    
class Evaluator(object):
    def __init__(self, eval_list, eval_path, **kwargs):
        self.eval_path = eval_path
        self.eval_list = eval_list

    def eval(self, model):
        model.eval()
        files = []
        embeddings = {}
        lines = open(self.eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm(enumerate(setfiles), desc='test', total=len(setfiles), ncols=100):
            audio, _ = soundfile.read(os.path.join(self.eval_path, file))
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float32)
            data_2 = torch.FloatTensor(feats).cuda()
            with torch.no_grad():
                embedding_1 = model.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = model.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF_05, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF_05


class FastTestDataset(Dataset):
    def __init__(self, eval_list, eval_list1, eval_list2, eval_path, **kwargs):        
        self.data_list, self.data_length = [], []
        self.eval_path = eval_path
        if os.path.exists(eval_list):
            lines = open(eval_list).read().splitlines()
            for line in tqdm(lines, desc=f'Read test list', ncols=120, total=len(lines)):
                self.data_list.append(line.split()[0])
                self.data_length.append(float(line.split()[1]))
        else:
            lines1 = open(eval_list1).read().splitlines()
            lines2 = open(eval_list2).read().splitlines()
            lines = lines1 + lines2
            lines = list(set(lines))
            for line in tqdm(lines, desc=f'Read test list', ncols=120, total=len(lines)):
                data = line.split()
                if data[1] not in self.data_list:
                    self.data_list.append(data[1])
                    audio, sr = soundfile.read(os.path.join(eval_path, data[1]))
                    data_length = len(audio) / sr
                    self.data_length.append(data_length)
                if data[2] not in self.data_list:
                    self.data_list.append(data[2])
                    audio, sr = soundfile.read(os.path.join(eval_path, data[2]))
                    data_length = len(audio) / sr
                    self.data_length.append(data_length)
            with open(eval_list, 'w') as f:
                for i in range(len(self.data_list)):
                    f.write(f"{self.data_list[i]} {self.data_length[i]:.2f}\n")
                print(f"Write {eval_list} done!")
                f.close()
        
        self.minibatch = []
        inds = numpy.array(self.data_length).argsort()
        self.data_list, self.data_length = numpy.array(self.data_list)[inds], \
                                            numpy.array(self.data_length)[inds]
        start = 0
        minibatch_size = 10
        while True:
            frame_length = self.data_length[start]
            end = min(len(self.data_list), start + minibatch_size)
            self.minibatch.append([self.data_list[start:end], frame_length])
            if end == len(self.data_list):
                break
            start = end
        print(f"Fast Test Dataset: {len(self.data_list)} samples, {len(self.minibatch)} minibatches")
        

    def __getitem__(self, index):
        data_lists, frame_length = self.minibatch[index]

        filenames, segments = [], []

        for num in range(len(data_lists)):
            file_name = data_lists[num]
            filenames.append(file_name)

            audio, sr = soundfile.read(os.path.join(self.eval_path, file_name))
            if len(audio) < int(frame_length * sr):
                shortage    = int(frame_length * sr) - len(audio) + 1
                audio       = numpy.pad(audio, (0, shortage), 'wrap')
            audio = numpy.array(audio[:int(frame_length * sr)])
            segments.append(audio)
    
        segments = torch.FloatTensor(numpy.array(segments))
        return segments, filenames

    def __len__(self):
        return len(self.minibatch)


class FastEvaluator(object):
    def __init__(self, eval_list, eval_list1, eval_list2, eval_path, **kwargs):
        self.eval_list = eval_list
        self.eval_list1 = eval_list1
        self.eval_list2 = eval_list2
        self.eval_path = eval_path
        self.test_dataset = FastTestDataset(eval_list, eval_list1, eval_list2, eval_path)
        self.test_loader = DataLoader(self.test_dataset, batch_size = 1, shuffle = False, num_workers = 10, drop_last = False)

    def eval(self, model):
        model.eval()
        embeddings = {}
        for data, filenames in tqdm(self.test_loader, desc='extract speaker embedding', total = len(self.test_loader), ncols=120):
            with torch.no_grad():
                embedding = model.speaker_encoder.forward(data[0].cuda(), aug=False)
                for num in range(len(filenames)):
                    filename = filenames[num][0]
                    a = torch.unsqueeze(embedding[num], dim = 0)
                    embeddings[filename] = F.normalize(a, p=2, dim=1)
        
        results = []
        if self.eval_list1 == self.eval_list2:
            task_list = [self.eval_list1]
        else:
            task_list = [self.eval_list1, self.eval_list2]
        for eval_list in task_list:
            scores, labels = [], []
            lines = open(eval_list).read().splitlines()
            for line in lines:
                embedding_1 = embeddings[line.split()[1]]
                embedding_2 = embeddings[line.split()[2]]
                score = torch.mean(torch.matmul(embedding_1, embedding_2.T)).detach().cpu().numpy()
                scores.append(score)
                labels.append(int(line.split()[0]))

            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF_05, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

            if len(task_list) == 1:
                return EER, minDCF_05
            results.append([EER, minDCF_05])

        return results
