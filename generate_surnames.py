import csv
import re
from collections import defaultdict

# A comprehensive list of several hundred traditional South Indian surnames,
# heavily featuring multi-syllabic house names, patronymics, and titles.
names_data = [
    # Tamil (Patronymics, Deity names, Village names)
    "Balasubramanian", "Radhakrishnan", "Venkatraman", "Muthuswamy", "Ramamoorthy",
    "Krishnamoorthy", "Chandrasekaran", "Meenakshisundaram", "Thirunavukkarasu", "Annamalai",
    "Ponnambalam", "Sivasubramanian", "Kalyanasundaram", "Vijayaraghavan", "Janakiraman",
    "Gopalakrishnan", "Lakshminarayanan", "Srinivasan", "Parthasarathy", "Ranganathan",
    "Swaminathan", "Ramanathan", "Viswanathan", "Vaidyanathan", "Sethuraman",
    "Jeyaraman", "Natarajan", "Nagarajan", "Soundararajan", "Thiagarajan",
    "Rajagopalan", "Venkatachalam", "Arunachalam", "Satchidanandam", "Sabapathy",
    "Ganapathy", "Ramakrishnan", "Unnikrishnan", "Muthukrishnan", "Sivakumar",
    "Ramalingam", "Mahalingam", "Sivalingam", "Dhandapani", "Palanisamy",
    "Ramasamy", "Karuppasamy", "Madasamy", "Muniyandi", "Sudalaimuthu",
    "Sivaramakrishnan", "Ananthakrishnan", "Balakrishnan", "Jayaraman", "Karthikeyan",
    "Murugesan", "Paramasivam", "Rathinam", "Singaravelan", "Thangavelu",
    "Udhayakumar", "Venkatesan", "Vedanayagam", "Somasundaram", "Chidambaram",
    "Kathirgamanathan", "Anandasangaree", "Bahukutumbi", "Visweswaran", "Sivanandan",
    "Ramachandran", "Subramanian", "Arulpragasam", "Gounder", "Chettiar", "Mudaliar",

    # Telugu (Intiperu - House Names, Ancestral Villages)
    "Yelamanchili", "Kasinadhuni", "Bhagavatula", "Saraswatula", "Chaturvedula",
    "Samayamantri", "Dwibhashyam", "Tanguturi", "Mamidipudi", "Nandamuri",
    "Akkineni", "Daggubati", "Uppalapati", "Alluri", "Pingali", "Vemuri",
    "Kompella", "Vangapandu", "Gummadi", "Kambhampati", "Jonnalagadda",
    "Yarlagadda", "Pasupaleti", "Tallapaka", "Annamayya", "Mutnuri",
    "Bhandaru", "Chilakamarti", "Kandukuri", "Gurazada", "Rayaprolu",
    "Devulapalli", "Viswanatha", "Cingireddy", "Kalvakuntla", "Madabhushi",
    "Gottipati", "Duggirala", "Kodali", "Muppaneni", "Nalapati", "Penumarthi",
    "Ramineni", "Samineni", "Tummala", "Vemulapalli", "Yenduri", "Zampala",
    "Bhattiprolu", "Cherukuri", "Dandamudi", "Gudivada", "Kakarla", "Maddineni",
    "Nadendla", "Parvathaneni", "Ravipati", "Sivaratri", "Tenali", "Vallabhaneni",
    "Appikatla", "Bhimavaram", "Chodavarapu", "Dhulipala", "Gollapudi", "Inampudi",
    "Jamalapuram", "Kanneganti", "Lankapalli", "Machilipatnam", "Nidamanuru", "Oruganti",
    "Paladugu", "Rentachintala", "Sirikonda", "Tirumalasetti", "Vajhala", "Yellapragada",

    # Malayalam (Tharavadu - Ancestral Homes, Compounds)
    "Puthenpurayil", "Kizhakkethil", "Thekkedathu", "Kunnumpurathu", "Parambil",
    "Vadakkedathu", "Padinjarekara", "Peediyakkal", "Thottumkal", "Kanjirappally",
    "Peringarappillil", "Chakyarampurathu", "Namboothiripad", "Kizhakkemadam", "Thekkemadam",
    "Ponneth", "Valiyaparambil", "Kochuparambil", "Pulikkal", "Karingal",
    "Muttathiparambil", "Kizhakkepattu", "Pazhampillil", "Elanjikkal", "Kallumkal",
    "Manjaly", "Vattakuzhy", "Tharayil", "Kalarikkal", "Panamoottil",
    "Kanjirakkattu", "Thoppil", "Nedumparambil", "Kuttikkat", "Puthuppally",
    "Maliyekkal", "Kottarakkara", "Mavelikkara", "Thiruvananthapuram", "Pathanamthitta",
    "Cherthala", "Ambalappuzha", "Kuttanad", "Changanassery", "Meenachil",
    "Vaikom", "Muvattupuzha", "Kothamangalam", "Aluva", "Paravur",
    "Kodungallur", "Chalakudy", "Mukundapuram", "Thalappilly", "Chittur",
    "Palakkad", "Mannarkkad", "Ottapalam", "Perinthalmanna", "Tirur",
    "Ponnani", "Kozhikode", "Vadakara", "Thalassery", "Kannur",
    "Taliparamba", "Payyanur", "Kasaragod", "Hosdurg", "Kunjipurayil",
    "Peechamveettil", "Puthenveettil", "Vadakkepurayil", "Padinjareveettil", "Kizhakkepurayil",

    # Kannada (Villages, Professions, Matha associations)
    "Hiremath", "Chikkamath", "Doddagoudar", "Chikkaballapur", "Somayyaji",
    "Upadhyaya", "Thamraparni", "Deshpande", "Kulkarni", "Byndoor",
    "Kundapur", "Mangaluru", "Shivamogga", "Chitradurga", "Davanagere",
    "Hosapete", "Nanjangud", "Srirangapatna", "Basavanagudi", "Malleshwaram",
    "Padmanabhanagar", "Jayanagar", "Vidyaranyapura", "Yelahanka", "Bommanahalli",
    "Kengeri", "Rajarajeshwari", "Mahalakshmi", "Yeshwanthpur", "Basaveshwara",
    "Siddaramaiah", "Yediyurappa", "Kumaraswamy", "Devegowda", "Siddaraju",
    "Nanjundaswamy", "Byregowda", "Chaluvarayaswamy", "Revanna", "Bhavikatti",
    "Kallimani", "Huded", "Gaddigoudar", "Kattimani", "Nadig", "Nadagouda",
    "Patil", "Joshi", "Bhat", "Hegde", "Rao", "Nayak", "Shetty", "Gowda",
    "Hiremathadakatti", "Doddaballapura", "Ramanagara", "Chamarajanagar", "Mandya",
    "Hassan", "Tumakuru", "Kolar", "Chikkamagaluru", "Udupi",
    "Uttarakannada", "Dharwad", "Belagavi", "Vijayapura", "Bagalkot",
    "Raichur", "Koppal", "Gadag", "Haveri", "Ballari"
]

# Ensure uniqueness
names_data = list(set(names_data))

def count_syllables(word):
    word = word.lower()
    if not word: return 0
    
    vowels = 'aeiouy'
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
        
    # Adjustments
    if word.endswith('e') and not word.endswith('ee'):
        pass # e is pronounced in these languages
    if 'ia' in word: count += 1
    if 'io' in word: count += 1
    if 'iy' in word and len(word) > 2: count += 1
    
    # Overrides for short/common names
    overrides = {
        'rao': 1, 'bhat': 1, 'nair': 1, 'hegde': 2, 'gowda': 2, 'patil': 2,
        'joshi': 2, 'shetty': 2, 'nayak': 2
    }
    return overrides.get(word, count)

syllable_counts = defaultdict(int)
name_lengths = []

with open('dravidian_surnames_large.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Surname', 'Estimated_Syllables'])
    
    for name in names_data:
        syl = count_syllables(name)
        syllable_counts[syl] += 1
        writer.writerow([name, syl])
        name_lengths.append(syl)

print(f"Total Unique Names Generated: {len(names_data)}")
print("\n--- Syllable Breakdown ---")
for count in sorted(syllable_counts.keys()):
    print(f"{count} Syllables: {syllable_counts[count]} names")

avg_syl = sum(name_lengths) / len(name_lengths)
print(f"\nAverage Syllables per Name: {avg_syl:.2f}")
