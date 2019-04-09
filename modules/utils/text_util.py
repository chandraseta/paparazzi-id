import re

STOPWORDS_ID_FILEPATH = 'data/indo_stopwords.txt'


def load_stopwords(stopwords_filepath: str = STOPWORDS_ID_FILEPATH) -> [str]:
    sw_file = open(stopwords_filepath)
    stopwords = []

    for line in sw_file:
        stopwords.append(line.strip())

    return stopwords


def split_to_sentences(text: str) -> [str]:
    # Convert . -> $$ inside parenthesis and quotes
    text = re.sub('\.(?=[^(]*\\))', '$$', text)
    text = re.sub(r"(?<=([\"]\b))(?:(?=(\\?))\2.)*?(?=\1)", lambda x:x.group(0).replace('.', '$$'), text)

    sentences = text.split('. ')
    temp_sentence = ''
    return_sentences = []

    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()

        if sentence != '':
            if temp_sentence == '':
                temp_sentence = sentence
            else:
                temp_sentence += '. ' + sentence

            # Check if the first word in next sentence begins with uppercase letter
            next_sentence = '\\'
            if idx+1 < len(sentences):
                if sentences[idx+1] != '':
                    next_sentence = sentences[idx+1]

            # Also check if last word is less than 3 character
            if len(sentence.split(' ')[-1]) < 3 or next_sentence == '\\' or next_sentence[0] != next_sentence[0].upper():
                continue
            else:
                # Convert back $$ -> .
                temp_sentence = temp_sentence.replace('$$', '.')

                # Add period
                temp_sentence += '.'

                # Export sentence
                if len(temp_sentence) >= 20:

                    # Remove multiple periods
                    temp_sentence = re.sub('\.+$', '.', temp_sentence)

                    # Remove multiple whitespaces
                    temp_sentence = re.sub('\s+', ' ', temp_sentence).strip()
                    return_sentences.append(temp_sentence)

                temp_sentence = ''

    return return_sentences


{"category": "tajuk utama",
 "gold_labels": [
     [true],
     [false],
     [false],
     [false, false],
     [true],
     [false, false],
     [false],
     [false],
     [false],
     [false],
     [false],
     [false, false],
     [false],
     [false, false]],
 "id": "1519679780-demokrasi-makin-transaksional-ketua-dpr-galau",
 "paragraphs": [
     [
         ["Suara.com", "-", "Ketua", "DPR", "RI", "Bambang", "Soesatyo", "mengungkapkan", "kegalauannya", "melihat", "perkembangan", "demokrasi", "di", "Indonesia", "yang", "makin", "mengarah", "pada", "demokrasi", "transaksional", "karena", "berpotensi", "mengancam", "independensi", "bangsa", "Indonesia", "."]
     ],
     [
         ["Bambang", "Soesatyo", "mengatakan", "hal", "itu", "dalam", "sambutannya", "ketika", "menghadiri", "acara", "peresmian", "Grha", "Suara", "Muhammadiyah", ",", "di", "Yogyakarta", ",", "Minggu", "(", "25", "/", "2", "/", "2018", ")", ",", "seperti", "dikutip", "melalui", "siaran", "persnya", "."]
     ],
     [
         ["Hadir", "dalam", "acara", "tersebut", "antara", "lain", ",", "Ketua", "Umum", "PP", "Muhammadiyah", "Haedar", "Nashir", ",", "mantan", "Ketua", "Umum", "PP", "Muhammadiyah", "Buya", "Ahmad", "Syafii", "Maarif", ",", "Menkominfo", "Rudiantara", ",", "Mendikbud", "Muhadjir", "Effendy", ",", "anggota", "Fraksi", "Partai", "Golkar", "DPR", "RI", "Mukhamad", "Misbakhun", ",", "anggota", "Fraksi", "Nasdem", "DPR", "RI", "Ahmad", "Syahroni", ",", "Ketua", "Umum", "PP", "Aisyah", "Nurjanah", ",", "Kapolda", "DI", "Yogyakarta", "Brigjen", "Pol", "Ahmad", "Dofiri", ",", "serta", "mantan", "Ketua", "KPK", "Busyro", "Muqoddas", "."]], [["\"", "Saya", "secara", "khusus", "meminta", "kepada", "Muhammadiyah", "untuk", "mengkaji", "kembali", "sistem", "pemilihan", "langsung", "dalam", "demokrasi", "kita", ",", "terutama", "dalam", "pilkada", "."], ["Apakah", "lebih", "banyak", "manfaatnya", "atau", "mudaratnya", "bagi", "rakyat", ",", "\"", "kata", "Bambang", "Soesatyo", "."]], [["Menurut", "Bambang", ",", "demokrasi", "transaksional", "yang", "mulai", "tidak", "terkendali", "jika", "terus", "ini", "dibiarkan", ",", "maka", "bukan", "tidak", "mungkin", "suatu", "saat", "Indonesia", "akan", "dikuasai", "para", "pemilik", "modal", ",", "baik", "secara", "langsung", "maupun", "tidak", "langsung", "."]], [["\"", "Bisa", "jadi", "pada", "10", "tahun", "atau", "20", "tahun", "ke", "depan", ",", "Indonesia", "tidak", "lagi", "memiliki", "presiden", "yang", "akhiran", "namanya", "huruf", "O", ",", "eperti", "Soekarno", ",", "Soeharto", ",", "Susilo", "Bambang", "Yudhoyono", ",", "dan", "Joko", "Widodo", "."], ["Karena", "peran", "para", "pemilik", "modal", "semakin", "mendominasi", ",", "\"", "tegasnya", "."]], [["Pada", "acara", "tersebut", ",", "Bamsoet", ",", "panggilan", "Bambang", "Soesatyo", "juga", "secara", "sepontan", "mendapat", "penghargaan", "sebagai", "warga", "Muhammadiyah", "yang", "ditandai", "dengan", "pekamaian", "baju", "batik", "Muhammadiyah", "berwarna", "hijau", "dan", "syal", "warna", "merah", "bertuliskan", "\"", "Suara", "Muhammadiyah", "\"", "."]], [["Bamsoet", "diberikan", "penghargaan", "sebagai", "warga", "Muhammadiyah", "karena", "pemikiran", "serta", "visi", "dan", "misinya", "dinilai", "sejalan", "dengan", "visi", "dan", "misi", "Muhammadiyah", "."]], [["\"", "Saya", "bangga", "menjadi", "bagian", "dari", "keluarga", "besar", "Muhammadiyah", ",", "walaupun", "baru", "hari", "ini", "saya", "dipakaikan", "baju", "batik", "Muhammadiyah", ",", "\"", "katanya", "."]], [["Kemudian", ",", "perihal", "peresmian", "gedung", "Grha", "Suara", "Muhammadiyah", ",", "Bamsoet", "berharap", "sarana", "itu", "makin", "memajukan", "Suara", "Muhammadiyah", "sebagai", "media", "kebanggaan", "dari", "seluruh", "organisasi", "kemasyarakatan", "yang", "didirikan", "KH", "Ahmad", "Dahlan", "."]], [["Bamsoet", "yang", "pernah", "berprofesi", "sebagai", "wartawan", "menambahkan", ",", "tantangan", "bagi", "bisnis", "media", "di", "era", "digital", "saat", "ini", "makin", "berat", "."]], [["\"", "Media", "konvensional", "tidak", "sedikit", "yang", "gulung", "tikar", "karena", "tidak", "dapat", "menyesuaikan", "perubahan", "zaman", "."], ["Saya", "angkat", "topi", "karena", "Suara", "Muhammadiyah", "yang", "sudah", "berusia", "103", "tahun", "tetap", "bertahan", "dan", "menjadi", "media", "terlama", "yang", "masih", "terbit", ",", "\"", "ujar", "Bamsoet", "."]], [["Bamsoet", "juga", "berpesan", "agar", "Suara", "Muhammadiyah", "tidak", "berhenti", "berikhtiar", "dan", "melakukan", "terobosan", "untuk", "pembangunan", "bangsa", "."]], [["Apalagi", ",", "Suara", "Muhammadiyah", "juga", "menerima", "penghargaan", "sebagai", "Media", "Dakwah", "Perjuangan", "Kemerdekaan", "RI", "dalam", "Bahasa", "Indonesia", ",", "pada", "Hari", "Pers", "Nasional", "2018", "."], ["(", "Antara", ")"]]], "source": "suara", "source_url": "https://www.suara.com/news/2018/02/26/060000/demokrasi-makin-transaksional-ketua-dpr-galau", "summary": [["Ketua", "DPR", "RI", ",", "Bambang", "Soesatyo", ",", "mengungkapkan", "kegalauannya", "melihat", "perkembangan", "demokrasi", "di", "Indonesia", "yang", "makin", "mengarah", "pada", "demokrasi", "transaksional", "karena", "berpotensi", "mengancam", "independensi", "bangsa", "Indonesia", "."], ["Menurut", "Bambang", ",", "demokrasi", "transaksional", "yang", "mulai", "tidak", "terkendali", "jika", "terus", "ini", "dibiarkan", ",", "maka", "bukan", "tidak", "mungkin", "suatu", "saat", "Indonesia", "akan", "dikuasai", "para", "pemilik", "modal", ",", "baik", "secara", "langsung", "maupun", "tidak", "langsung", "."]]}
