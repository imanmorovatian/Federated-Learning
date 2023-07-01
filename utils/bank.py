import numpy as np
from utils.utils import extract_amplitude


class Bank:

    def __init__(self, clients):
        self.clients = clients
        self.bank = {}

        # bank = {
        #   client 1 = {
        #       label 1 = []
        #       label 2 = []
        #       ...     
        #   }
        #   
        #  client 2 = {
        #       label 1 = []
        #       label 2 = []
        #       ...
        #   }
        #
        #  ...
        # }

        for client in self.clients:
            client_amps = {}
            for image, label in client.dataset:
                amp = extract_amplitude(image)
                client_amps.setdefault(label, [])
                client_amps[label].append(amp)
            
            self.bank[client.name] = client_amps


    def get_amps(self, client_name, image_label, other_client_names):
        image_label = int(image_label)
        collected_amps = []
        for client, amps in self.bank.items():
            if client != client_name and client in other_client_names:
                if image_label in amps.keys():
                    if len(amps[image_label]) > 1:
                        all_indices = range(len(amps[image_label]))
                        index = np.random.choice(all_indices, 1)[0]
                        sample = amps[image_label][index]
                    else:
                        sample = amps[image_label][0]
                    collected_amps.append(sample)


        return  collected_amps
