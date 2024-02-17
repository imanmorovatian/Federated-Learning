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
        alternative_amps = []
        limit = len(other_client_names)

        for name in other_client_names:
            if name != client_name:
                if image_label in self.bank[name].keys():
                    if len(self.bank[name][image_label]) > 1:
                        all_indices = range(len(self.bank[name][image_label]))
                        index = np.random.choice(all_indices, 1)[0]
                        sample = self.bank[name][image_label][index]
                    else:
                        sample = self.bank[name][image_label][0]

                    collected_amps.append(sample)

        
        if len(collected_amps) == 0:
            for client, amps in self.bank.items():
                if len(alternative_amps) <= limit and client != client_name:
                    if image_label in amps.keys():
                        if len(amps[image_label]) > 1:
                            all_indices = range(len(amps[image_label]))
                            index = np.random.choice(all_indices, 1)[0]
                            sample = amps[image_label][index]
                        else:
                            sample = amps[image_label][0]

                        alternative_amps.append(sample)


            return  alternative_amps
        else:
            return collected_amps
