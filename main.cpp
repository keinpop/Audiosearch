#include <iostream>
#include<fstream>
#include <vector>
#include <complex>
#include <iomanip>
#include <algorithm>

#define MINIMP3_IMPLEMENTATION

#include "minimp3.h"
#include "minimp3_ex.h"

const int WIDTH_S = 4096;
const int STEP_S = 1024;

const int COUNT_OF_OCTAVES = 7;

const int EXPERIMENTAL_DIFF = 7;
const int MAX_DIFF_COUNT = 3;

const double LOWER_LIMIT_COEFF = 0.4;
/*
    данная константа нам нужна для того, чтобы была некая точка останова 
    для бинарного поиска по промежуткам. Тк при 2^12 == 4096 уже значение 
    коэффициента меньше 0,4 => нам уже не интересно это. 
*/
const int MAX_POW_OF_TWO = 12;  

const double PI = std::acos(-1);

const std::vector<int> OCTAVES = {0, 32, 64, 128, 256, 512, 1024, 2048};

struct Imprint 
{
    short v[COUNT_OF_OCTAVES];

    void SlicesFrequency(std::vector< std::complex <double>> & spec) {
        for (int i = 0; i < COUNT_OF_OCTAVES; ++i) {
            double max = 0;
            short freq;

            for (int j = OCTAVES[i]; j < OCTAVES[i + 1]; j++) {
                if (std::abs(spec[j]) > max) {
                    max = std::abs(spec[j]);
                    freq = j;
                }
            }

            this->v[i] = freq;
        }
    }
};

double HAMMING_WINDOW(int k) {return (0.5 * (1 - std::cos((2 * PI * (k)) / (WIDTH_S - 1))));}

std::vector<short> decoder(std::string filename) 
{
    mp3dec_t mp3d;
    mp3dec_file_info_t info;
    if (mp3dec_load(&mp3d, filename.c_str(), &info, NULL, NULL)) {
        throw std::runtime_error("Decode error - file "+filename);
    }
    std::vector<short> file(info.buffer, info.buffer + info.samples);
    free(info.buffer);
    std::vector <short> res;
    for (int i = 0; i < file.size(); i += info.channels) {
        int sum = 0;
        for (int j = 0; j < info.channels; j++) {
            sum += file[i + j];
        }

        res.push_back(sum / info.channels);
    }

    return res;
}

int ReverseBitsIndex(int size, int idxV) 
{
    int reverse = 0;

    while (size > 1) {
        reverse |= (idxV & 1); // добавляем младший бит к реверсу
        reverse <<= 1; // сдвигаем влево на 1 бит
        idxV >>= 1; // сдвигаем вправо, чтобы некст бит был младшим
        size /= 2; // уменьшаем size для некст операции   
    }

    reverse >>= 1; // компенсируем лишний сдвиг

    return reverse;
}

void FastFourierTransform(
    const std::vector<double> & inp, 
    std::vector< std::complex< double>> & res,
    std::complex<double> phaseRoot,
    int l,
    int r
) 
{
    // Единственный элемент в диапазоне
    if (r == l) {
        int reverseIdx = ReverseBitsIndex(inp.size(), l);
        res[r] = inp[reverseIdx]; 
        return;
    }
    // Делим на две части (разделяй и властвуй)
    int midd = (l + r) / 2;
    // И рекурсивно запускаем тоже самое для двух частей
    FastFourierTransform(inp, res, std::pow(phaseRoot, 2), l, midd);
    FastFourierTransform(inp, res, std::pow(phaseRoot, 2), midd+1, r);

    // Вычислим фазовый множитель и объединение результатов
    int rangeSize = r - l + 1;
    std::vector< std::complex< double>> tmp(rangeSize);
    int halfRS = rangeSize / 2;
    std::complex<double> phaseMultiplier = 1.0;

    // Комбинируем результаты для левой и правой части
    for (int i = 0; i < rangeSize; ++i) {
        tmp[i] = res[l + i % halfRS] + phaseMultiplier * res[midd + 1 + i % halfRS];
        phaseMultiplier *= phaseRoot;
    }

    // копируем временный результат обратно в выходной массив 
    for (int i = 0; i < rangeSize; ++i) {
        res[l + i] = tmp[i];
    }
}

std::vector<Imprint> ComputeResSpectrum(std::vector<short> sample)
{
    std::vector<Imprint> ans;
    for (size_t i = 0; i < sample.size() - WIDTH_S; i += STEP_S) {
        std::vector<double> window(WIDTH_S);
        std::vector< std::complex< double>> res(WIDTH_S);

        for (int j = 0; j < WIDTH_S; ++j) {
            window[j] = sample[i + j] * HAMMING_WINDOW(j);
        }

        int winSize = window.size();

        std::complex<double> compFromPolar = std::polar((double)1.0, 2 * PI / winSize);

        FastFourierTransform(window, res, compFromPolar, 0, winSize - 1);

        Imprint imp;
        imp.SlicesFrequency(res);

        ans.push_back(imp);
    }

    return ans;
}

class AudioSearch 
{
public:
    AudioSearch(std::vector<short> samples);
    AudioSearch(std::ifstream & in);
    
    void Write(std::ofstream & out);

    // set&get
    std::string GetTitle();
    void SetTitle(std::string t);

    std::vector<Imprint> GetData();

    size_t GetDataSize();

    std::vector<Imprint> Data;
private:
    std::string Title;
}; // AudioSearch

AudioSearch::AudioSearch(std::vector<short> samples) 
{
    Data = ComputeResSpectrum(samples);
}

AudioSearch::AudioSearch(std::ifstream & in)
{
    in >> Title;
    int sz;
    in >> sz; 
    for (int i = 0; i < sz; i++) {
        Imprint tmp;
        for (int j = 0; j < COUNT_OF_OCTAVES; j++) {
            in >> tmp.v[j];
        }

        Data.push_back(tmp);
    }
}

void AudioSearch::Write(std::ofstream & out) 
{
    out << Title << '\n';
    out << Data.size() << '\n';

    for (int i = 0; i < Data.size(); ++i) {
        for (int j = 0; j < COUNT_OF_OCTAVES; ++j) {
            out << Data[i].v[j] << " ";
        }

        out << '\n';
    }
}

std::string AudioSearch::GetTitle()
{
    return Title;
}
    
void AudioSearch::SetTitle(std::string t)
{
    Title = t;   
}

size_t AudioSearch::GetDataSize()
{
    return Data.size();
}

std::vector<Imprint> AudioSearch::GetData()
{
    return this->Data;
}

int CompareImprintsData(Imprint a, Imprint b)
{
    int countMatch = 0;

    for (int i = 0; i < COUNT_OF_OCTAVES; i++) {
        if (abs(a.v[i] - b.v[i]) < EXPERIMENTAL_DIFF) {
            ++countMatch;
        }
    }

    // если превышаем количества различий
    if (countMatch >= MAX_DIFF_COUNT) {
        return 0;
    } else {
        // такое значение в вагнере-фишере
        return 1;
    }

    // заглушка
    return -1;
}

int LevenshteinLen(int pos, AudioSearch & a, AudioSearch & b)
{
    int size = std::min(a.GetDataSize(), b.GetDataSize());

    std::vector< std::vector <int>> dp(size + 1, std::vector<int>(size + 1));

    // base dp
    dp[0][0] = 0;
    for (int i = 0; i <= size; i++) {
        // особенность алгоритма
        dp[i][0] = i;
        dp[0][i] = i;
    }

    for (int i = 1; i <= size; i++) {
        for (int j = 1; j <= size; j++) {
            // у нас есть 3 проверки
            int first = dp[i - 1][j] + 1;
            int second = dp[i][j - 1] + 1;
            int third = dp[i - 1][j - 1] + CompareImprintsData(a.Data[pos + i - 1], b.Data[j - 1]);

            dp[i][j] = std::min(first, std::min(second, third));
        }
    }

    return dp[size][size];
}

// функция подсчета нашего коэффициента схожести
double GetCoeffIdentically(int pos, AudioSearch & fullAudio, AudioSearch & sample)
{
    int minSize = std::min(fullAudio.GetDataSize(), sample.GetDataSize());

    // коэфф, который мы вычитаем из единицы логичен, тк значение
    // Левенштейна у нас максимум может быть длиной большего "слова"
    // мы делим на 2 * минСайз, тк при  LevLen равному 1 +- 2, совпадение 
    // явно большое, а значит мы нашли совпадение
    return 1 - (LevenshteinLen(pos, fullAudio, sample) / (2.0 * minSize));
}

double Search(AudioSearch & fullAudio, AudioSearch & sample)
{
    int faSize = fullAudio.GetDataSize();
    int sampSize = sample.GetDataSize();

    double ans = 0; // максимальное совпадение

    /*
        Идея - идти по треку "кусками" сэмпла, и глядеть на схожесть,
        но тк у нас бывают ситуации, када отрывок может быть не ровно 
        там, как мы сравниваем, а например "съехать" левее/правее, поэтому
        будем смотреть совпадение двух соседних участков
    */

    double prevCoeff = GetCoeffIdentically(0, fullAudio, sample);
    for (int i = sampSize; i < faSize - sampSize; i += sampSize) {
        double currCoeff = GetCoeffIdentically(i, fullAudio, sample);

        if (prevCoeff + currCoeff < LOWER_LIMIT_COEFF) {
            prevCoeff = currCoeff;
            continue;
        }

        int l = i - sampSize;
        int r = i + 1;

        int lCoeff = prevCoeff;
        int rCoeff = currCoeff;

        int j = MAX_POW_OF_TWO;
        while (j > 0) {
            j--;
            int midd = (l + r) / 2;

            double middCoeff = GetCoeffIdentically(midd, fullAudio, sample);
            ans = std::max(ans, middCoeff);
            
            if (lCoeff > rCoeff) {
                r = midd;
                rCoeff = middCoeff;
            } else if (lCoeff <= rCoeff) {
                l = midd;
                lCoeff = middCoeff;
            }
        }

        prevCoeff = currCoeff;
    }

    return ans;
}

std::string ARG_INPUT {"--input"};
std::string ARG_OUTPUT {"--output"};
std::string ARG_INDEX {"--index"};

std::string CMD_INDEX {"index"};
std::string CMD_SEARCH {"search"};

const int NOT_FOUND_IDX = -1;

// g++ -std=c++11 main.cpp

/*
    ./a.out index --input tracks.txt \
                --output lib.txt

*/
int main(int argc, char* argv[])
{
    std::string cmd { argv[1] };
    
    std::ifstream inFile;
    std::ofstream outFile;

    std::ifstream idxFile;

    std::vector<std::string> inpFiles;

    for (int i = 2; i < argc; ++i) {
        if (argv[i] == ARG_INPUT) {
            inFile = std::ifstream(argv[i+1]);
            while(!inFile.eof()) {
                std::string curr;
                std::getline(inFile, curr);
                inpFiles.push_back(curr);
            }
        } else if (argv[i] == ARG_OUTPUT) {
            outFile = std::ofstream(argv[i + 1]);
        } else if (argv[i] == ARG_INDEX) {
            idxFile = std::ifstream(argv[i + 1]);
        }
    }
    
    if (cmd == CMD_INDEX) {
        outFile << inpFiles.size() << '\n';

        for (int i = 0; i < inpFiles.size(); i++) {
            std::vector<short> samples = decoder(inpFiles[i]);
            AudioSearch as(samples);
            as.SetTitle(inpFiles[i]);
            as.Write(outFile);
        }
    } else if (cmd == CMD_SEARCH) {
        int countOfAudio;
        idxFile >> countOfAudio;

        std::vector<AudioSearch> audios;

        for (int i = 0; i < countOfAudio; i++) {
            AudioSearch tmp(idxFile);
            audios.push_back(tmp);
        }

        for (int i = 0; i < inpFiles.size(); i++) {
            int maxMatchIdx = NOT_FOUND_IDX;
            std::vector<short> samples = decoder(inpFiles[i]);

            AudioSearch samp(samples);

            // дал возможность небольшой погрешности в виде 0.025
            double maxCoeff = LOWER_LIMIT_COEFF - 0.025;

            for (int j = 0; j < countOfAudio; j++) {
                double res = Search(audios[j], samp);
                
                if (res > maxCoeff) {
                    maxCoeff = res;
                    maxMatchIdx = j;
                } 
            }

            if (maxMatchIdx == NOT_FOUND_IDX) {
                outFile << "! NOT FOUND\n";
            } else {
                // напишем удовлетворяющий для тз вывод
                outFile << audios[maxMatchIdx].GetTitle() << '\n';

                // логируем с каким коэффом оно совпало
                std::cout << 
                    audios[maxMatchIdx].GetTitle() << 
                    " совпал с коэффициентом точности [" <<
                    maxCoeff << "] - " << inpFiles[i] << '\n';
            }
        }
    }

    return 0;
}
