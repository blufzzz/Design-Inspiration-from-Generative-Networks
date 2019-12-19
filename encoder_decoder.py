from torch import nn

class MaskEncoder(nn.Module):
    def __init__(self, dim, C = 2, multiplier = 32):
        super().__init__()
        self.cnn1 = nn.Conv2d(1,C*multiplier, kernel_size=(5,5), stride=(2,2))
        self.bn1 = nn.BatchNorm2d(C*multiplier)
        self.activation = nn.ReLU()

        self.cnn2 = nn.Conv2d(C*multiplier, 2*C*multiplier, kernel_size=(5,5), stride=(3,3))
        self.bn2 = nn.BatchNorm2d(2*C*multiplier)

        self.cnn3 = nn.Conv2d(2*C*multiplier, 4*C*multiplier, kernel_size=(5,5), stride=(2,2))
        self.bn3 = nn.BatchNorm2d(4*C*multiplier)
        
        self.cnn4 = nn.Conv2d(4*C*multiplier, 4*C*multiplier, kernel_size=(5,5), stride=(2,2))
        self.bn4 = nn.BatchNorm2d(4*C*multiplier)
        
        self.linear = nn.Linear(1024,dim)
    
    def forward(self, x):
        # x.shape is [bs,128,128,1]
        batch_size = x.shape[0]
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.activation(x)
        
        x = x.view(batch_size,-1)
        x = self.linear(x)
        
        return x
    
    
class MaskDecoder(nn.Module):
    def __init__(self, dim, C = 2, multiplier = 32):
        super().__init__()
        self.cnnt1 = nn.ConvTranspose2d(dim,4*C*multiplier, kernel_size=(4,4))
        self.bn1 = nn.BatchNorm2d(4*C*multiplier)
        self.activation = nn.ReLU()

        self.cnnt2 = nn.ConvTranspose2d(4*C*multiplier, 2*C*multiplier, kernel_size=(4,4), stride=(4,4))
        self.bn2 = nn.BatchNorm2d(2*C*multiplier)

        self.cnnt3 = nn.ConvTranspose2d(2*C*multiplier, C*multiplier, kernel_size=(4,4), stride=(4,4))
        self.bn3 = nn.BatchNorm2d(C*multiplier)
        
        self.cnnt4 = nn.ConvTranspose2d(C*multiplier, 1, kernel_size=(2,2), stride=(2,2))
    
    def forward(self, x):
        # x.shape is [bs,1024]
        batch_size, C = x.shape
        x = x.view(batch_size,C,1,1)
        x = self.cnnt1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.cnnt2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.cnnt3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        x = self.cnnt4(x)
        x = self.activation(x)
        
        return x    
