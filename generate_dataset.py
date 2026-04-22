import pandas as pd
import random

# ----------------------------
# CONFIG
# ----------------------------
NUM_COMMANDS = 3000
NUM_NEGATIVES = 400

# ----------------------------
# ACTIONS & DEVICES
# ----------------------------
actions = {
    "ENABLE": [
        "turn on",
        "switch on",
        "activate",
        "start",
        "enable",
        "power on",
        "on",
        "power on",
        "power up",
    ],
    "DISABLE": [
        "turn off",
        "switch off",
        "stop",
        "disable",
        "shut down",
        "power off",
        "off",
        "power down",
        "power off",
    ],
    "INCREASE": [
        "increase",
        "raise",
        "boost",
        "turn up",
        "make higher",
        "up",
        "turn up",
    ],
    "DECREASE": [
        "decrease",
        "lower",
        "reduce",
        "turn down",
        "make lower",
        "down",
        "turn down",
    ],
}

devices = {
    "LIGHTS": [
        "lights",
        "light",
        "lamp",
        "lighting",
        "bulb",
        "ceiling light",
        "room light",
        "bedside lamp",
    ],
    "AC": ["ac", "air conditioner", "aircon", "air conditioning", "cooler", "hvac"],
    "FAN": ["fan", "ceiling fan", "ventilator", "air fan"],
    "TV": ["tv", "television", "smart tv", "screen"],
    "MUSIC": ["music", "speaker", "audio", "sound system", "playlist", "spotify"],
}

# ----------------------------
# SPECIAL CASES
# ----------------------------
special_cases = [
    ("make it brighter", "INCREASE", "LIGHTS"),
    ("dim the lights", "DECREASE", "LIGHTS"),
    ("its too dim", "INCREASE", "LIGHTS"),
    ("make it colder", "DECREASE", "AC"),
    ("make it warmer", "INCREASE", "AC"),
    ("cool the room", "DECREASE", "AC"),
    ("heat up the room", "INCREASE", "AC"),
    ("its too cold", "INCREASE", "AC"),
    ("im cold", "INCREASE", "AC"),
    ("its hot", "INCREASE", "AC"),
    ("i am cold", "INCREASE", "AC"),
    ("i am freezing", "INCREASE", "AC"),
    ("its too hot", "DECREASE", "AC"),
    ("im hot", "DECREASE", "AC"),
    ("its hot", "DECREASE", "AC"),
    ("i am hot", "DECREASE", "AC"),
    ("its too dark", "INCREASE", "LIGHTS"),
    ("its too bright", "DECREASE", "LIGHTS"),
]

# ----------------------------
# NEGATIVE DATA
# ----------------------------
negative_templates = [
    "what is the {thing}",
    "tell me a {thing}",
    "can you {verb}",
    "i want to {verb}",
    "how do i {verb}",
    "{thing} is not working",
    "do you know how to {verb}",
]

manual_negatives = [
    "idk what im doing",
    "this is so stupid",
    "i am tired",
    "this is boring",
    "hello",
    "how are you",
    "what time is it",
    "im hungry",
    "this is annoying",
    "i dont like this",
    "tell me something interesting",
    "There is a dictionary open on the table.",
    "She studied music in college.",
    "You have to take the good with the bad.",
    "Thats why they pay us the big bucks.",
    "She was miserable the entire time she was pregnant.",
    "The tall girl tied her tennis shoes and went outside.",
    "Ill see to it.",
    "Its four a.m. and both of us would rather be asleep.",
    "The security is the safest around.",
    "She was in Australia when the COVID-lockdown started.",
    "I was excited to spend time with my wife without being interrupted by kids.",
    "Wiggle your fingers!",
    "I want to be done with this.",
    "Tulips grow every year.",
    "Im having second thoughts about getting married so soon.",
    "There are so many things to consider.",
    "Money doesnt grow on trees.",
    "This could become a problem.",
    "Robert tends to talk big.",
    "She bought the green eyeshadow spontaneously, then regretted the purchase.",
    "You did right in me by telling the truth.",
    "I wish I had met my uncle yesterday like I was supposed to.",
    "She raised her hand to ask a question.",
    "The volume was up too high on the television.",
    "Im going to Lollapalooza.",
    "This is too hefty to easily carry.",
    "I have a crush on Albert.",
    "He wanted to wear lipstick but the lipstick was expired.",
    "What a day were having! her mother sighed.",
    "The spoon lay next to the fork.",
    "This big sofa is really not suitable for a small room.",
    "Judith Jarvis Thompson is the best philosopher of all time.",
    "Today is the first day he made it to work on time in a while.",
    "I entered Toms name on the list of candidates.",
    "I feel like I am going to pass out.",
    "That was last week.",
    "This is the biggest apple Ive ever seen.",
    "Toms house is at least twice as big as mine.",
    "I wouldnt bet on going to the fair today.",
    "He continued talking.",
    "Its easier to clean with disposable dishes.",
    "Do you have any big plans for the weekend?",
    "My sister likes to eat cheese on her peanut butter sandwich and pickles on her ice cream.",
    "She had a hard time guessing what women were nannies and which ones were mothers themselves.",
    "Tom has a pain in his big toe on his right foot.",
    "We have a juice bar.",
    "Bumble is better.",
    "I have a friend whose father is the captain of a big ship.",
    "I had to put a pin in his femur.",
    "The broken leg isnt Toms biggest problem.",
    "In Australia, there is a store called PieFace which sells meat pies with smiley faces on them.",
    "Time is money.",
    "This is my big break.",
    "There used to be a big cherry tree behind my house.",
    "I am a genius.",
    "The giraffe licked the tree and spat at the gorilla.",
    "Tom stepped off the bus with a smile on his face.",
    "There was a fire last night.",
    "Give me the big knife to cut the bread.",
    "She stayed at a hostel in Poland which cost $13 a night and served free breakfast.",
    "Show me your rates, please.",
    "When you sit down for a long time, it hurts.",
    "It was a very cold fall.",
    "He took a thirty-minute shower, and by the time he stepped out, his fingers were as wrinkled as prunes.",
    "This way you can start to heal.",
    "That is fake news.",
    "The car crash was so instantaneous that I barely had time to process it.",
    "Old habits die hard.",
    "She waited all day for her boss to call, but heard nothing.",
    "He needed thirteen stitches.",
    "She went to the pier.",
    "Her water bottle had a built-in filtration system.",
    "Has she texted you back yet?",
    "They made crispy chicken.",
    "One cannot live on bread alone.",
    "Theres a big package on the desk for you.",
    "Whats your favorite holiday?",
    "When you enter a room, smile and say hello.",
    "She ate a sandwich for lunch which contained turkey, cheddar cheese, and slices of strawberries.",
    "Tom cant forget the time he and Mary had their first argument.",
    "Those apples are very red.",
    "Nikita is a perfectly respectable businessman.",
    "This room is just about big enough.",
    "I only like certain brands of clothes.",
    "I check off each task on my list as soon as I complete it.",
    "This problem is much bigger than we thought.",
    "I could hang out with her.",
    "Nothing beats a big glass of beer in summer.",
    "He was married to a friend of mine.",
    "he has skipped school on many occasions.",
    "The store had multiple skeletons they claimed were real, alongside a taxidermies, two-headed calf.",
    "Shape up or ship out!",
    "The bully at school was mean to everyone except me.",
    "The girl picked the pink flower out of her garden.",
    "That house is small, but its big enough for us.",
    "Its rare to find big yards in Japan.",
    "I will take you to the movies only if you wait for me outside.",
    "Im going to crash at your place.",
    "No woman would buy that.",
    "On the other hand, I feel like I just have to do it for my sanity.",
    "We should hang soon.",
    "He was dressed for work.",
    "She went to sleep after 12.",
    "He is capable of the crime.",
    "You had me worried for a moment.",
    "Its not the end of the world.",
    "She mixed up the recipes and accidentally made a beef parfait.",
    "He gets up early every day.",
    "Did you hear about the new animal discovered in Ecuador?",
    "How do you celebrate the 4th of July?",
    "A red tie will match that suit.",
    "I found a gold coin on the playground after school today.",
    "Once you join me, we can go visit the aquarium at the mall.",
    "He owes me a lot of money.",
    "There is a big stack of mail on the table.",
    "I hate cats even more than I hate dogs.",
    "I felt sort of sick.",
    "It will definitely happen sometime in the future.",
    "I can see you.",
    "If you dont shut up, I will turn this car around.",
    "I have been trying to say this for awhile.",
    "She was determined to summit Mount Everest before she turned twenty-nine.",
    "I made a list.",
    "Here is David with his guitar.",
    "She wished she had the budget to buy take-out, because she really wanted Thai food.",
    "The other baristas are so weird.",
    "Hes not the sharpest knife in the drawer.",
    "She had to start saving in October so she could afford to buy Christmas presents for all her family and friends.",
    "Her mom told her to calm down, but she couldnt relax.",
    "The smoke was high in the sky.",
    "Well put you on the list.",
    "I spread the map on the table.",
    "We have been waiting for this for awhile.",
    "The bigger boys torment the little ones.",
    "My big sister washes her hair every morning.",
    "What are you talking about?",
    "She didnt drink, but she didnt want people to realize that, so she ordered a ginger ale at the bar.",
    "She had strange, green and purple eyes.",
    "He won the race through his determination and focus.",
    "Your stance tells me.",
    "When she saw Walmart had rotisserie chickens, she thought, this is a miracle.",
    "Being fashionable is easy.",
    "I want many things for myself.",
    "The mailbox was bent and broken and looked like someone had knocked it over on purpose.",
    "You connect with nature, and whether you plant flowers or vegetables, you get great satisfaction once the fruits of your labor start appearing.",
    "My water bottle is white and made of steel.",
    "A big fire broke out after the earthquake.",
    "Being the last male on my dads side of the family has its perks.",
    "She liked him until he said he was a Marxist.",
    "He was such a beautiful man she reconsidered her current relationship.",
    "She found the necklace in a safe at the bottom of her parents closet.",
    "Did you open the door?",
    "She really wanted to go out to a diner and eat some greasy toast and eggs for breakfast.",
    "His mom was so upset that he knew he couldnt tell her the truth.",
    "The baby was so cute but she was crying so loud I had to plug my ears.",
    "I like veggies too.",
    "Your house is beautiful.",
    "The man looked at her strangely through the window.",
    "She found little pellets of iron in her garden bed.",
    "It was a tough time.",
    "There is nothing more that I need to hear.",
    "You can see the blood vessel on her eyelid.",
    "The plant was fake but looked very real.",
    "I need a dollar for the vending machine.",
    "In Guam, she laid in the sun and ate mangoes all day.",
    "She froze at the sight of the big spider.",
    "Tom doesnt want to make a big deal out of it.",
    "There were four people playing the game: her, her boyfriend, her boyfriends roommate, Steve, and Steves girlfriend, Cara.",
    "This will make a real difference.",
    "She wished she could speak Italian.",
    "He has a nice sum of money put away.",
    "We would move to a nicer house if we had more money.",
    "On top of all that, her parents told her yesterday that they were getting a divorce.",
    "The clock was ticking and kept me awake all night.",
    "When I went to the cabin up north, I had to bring a lot of board games to entertain myself.",
    "Weve got much bigger problems to deal with right now.",
    "It took quite a while.",
    "English Speaking Courses are the hardest ones.",
    "Could you tell me what time it is?",
    "Theres nobody out there.",
    "Her great-great-great grandfather was an outlaw in the Old West.",
    "She was very tired and frustrated.",
    "Youll move more than before.",
    "What kind of music do you like to listen to?",
    "I want racecars to be legal for average people.",
    "She started drinking again because she felt so awkward at social gatherings.",
    "The jar of candles was ready on the table.",
    "She hated chocolate chip cookies, but she loved gingersnaps.",
    "She left her umbrella at her dates house, and because the date was so bad, couldnt return to get it.",
    "What do you like to do with other people?",
    "She liked Dune, but it didnt pass the Bechdel Test.",
    "It looks like you have bigger problems.",
    "Who do they believe you are?",
    "I wanted a little more than that, to be perfectly honest, Trevor.",
    "Thats the biggest grasshopper Ive ever seen.",
    "He ran his horse up the hill.",
    "Blood is thicker than water.",
    "The man was busted for theft.",
    "You cant have your cake and eat it too.",
    "I really need to lose some weight.",
    "Happy Birthday to my 152 year old lover, Canada.",
    "Juice was something I never drank.",
    "Would you like a drink of my smoothie?",
    "Shes drop-dead gorgeous.",
    "She wanted to be a tutor but had a hard time getting started.",
    "When the cats away, the mice play.",
    "The sooner I get to bed the better.",
    "You can make ice cream.",
    "The cafe is empty aside from an old man reading a book about Aristotle.",
    "He invited me to dinner yesterday.",
    "I wanted to do well in the interview.",
    "The plant was dying a sad little death.",
    "The course starts next Sunday.",
    "Id like to talk about the benefits of having a pet pig.",
    "Its much bigger than I thought it would be.",
    "Im going to make you a list of what needs to be done.",
    "In Hong Kong, everyone lives stacked together.",
    "She was out on the deck of the ship when she saw the first ghosts.",
    "Dont get too excited!",
    "Trees are small when they are first planted.",
    "Do you recognize this woman?",
    "Whos calling on the phone this late at night?",
    "He read the report.",
    "Honesty is the best policy.",
    "I want you to remember that.",
    "Get away from me, you slimy little worm!",
    "You dont say.",
    "You guys like meat.",
    "This weekend is going to be the best weekend ever.",
    "People dont realize how often you can fail as a writer, even after you get a good job or get a book deal.",
    "It would be quite impossible to enumerate all the things in existence.",
    "Wheres the nearest store?",
    "The children are at home.",
    "He has made a big improvement in tennis.",
    "I go back to bedroom.",
    "Could you tell me the way to the station?",
    "Apple picking is not in season.",
    "He was innocent of the crime.",
    "My favorite one smells like roses.",
    "Kens uncle has a chicken farm.",
    "She constantly thought she was one mistake away from being fired.",
    "Are you really going to ignore me, after everything that happened yesterday?",
    "She started wondering if he was cheating on her when she saw the receipt for an expensive jewelry purchase.",
    "They painted the coffee shop ceiling black.",
    "She was curling her hair when she accidentally burned her forehead.",
    "She is my mother.",
    "The delayed train has been an unexpected event today.",
    "The new movie was not as good as I thought it would be.",
    "I have the flowers.",
    "Are you going to have a blue birthday cake for your next birthday?",
    "Heres my big brother. Doesnt he look good?",
    "I am completely drunk.",
    "How long does it really take to heal from a bullet wound?",
    "There was a big fire in my neighborhood.",
    "We can go to lunch.",
    "I love this headline so much.",
    "Hes got some brain condition that makes him think hes a flamingo.",
    "The kid had made it his mission to pee in every pool he swam in.",
    "I am about as tall as my father now.",
    "Thats a pretty big fish youve just caught.",
    "I have 10 hair straighteners.",
    "Mary enjoys cooking.",
    "Who wrote Treasure Island?",
    "We took refuge behind a tree.",
    "The laptop light was the only one light on the room.",
    "Is there a big market for this kind of thing these days?",
    "He never calls this late at night.",
    "A balanced diet is a cookie in each hand.",
    "I am afraid of dark afternoons.",
    "She sells Christmas trees.",
    "Workers will still have to work in order to live and pay their landlords for shelter.",
    "He had a sore throat, so I gave him my bottle of water.",
    "The library was quiet until we got here.",
    "I am the designated survivor.",
    "His voice was pitchy and weak, partially because he was so nervous.",
    "I have 60 pairs of shoes.",
    "That man is creepy.",
    "He asked if she wanted room for cream in her coffee.",
    "The planet hadnt yet been discovered.",
    "You dont need 20 captains.",
    "I prefer ice cream over cake.",
    "I have a brochure for our vacation.",
    "My only thing is that Im working from home tomorrow.",
    "Were over the hill now! he called over his shoulder.",
    "I cant swim after I drink milk.",
    "Can you come over now?",
    "You can never turn the clock back.",
    "Tom forgot his shopping list and only got half of what he wanted to.",
    "He got shot in the stomach, but he didnt die.",
    "My dad needs a new job.",
    "A yardstick is 3 feet long.",
    "Tom picked up his glass and took a big sip of wine.",
    "How big you are!",
    "Toms apartment is way too big for just one man.",
    "You guys can leave whenever you want.",
    "We saw her duck.",
    "Everything will be okay in the end.",
    "You can order it spicy or classic.",
    "Thats a wrap for me.",
    "We need congressional support.",
]


things = [
    "weather",
    "time",
    "news",
    "score",
    "temperature outside",
    "cup",
    "game",
    "tablet",
    "chips",
    "food",
    "gun",
]

verbs = [
    "play music",
    "open the door",
    "cook dinner",
    "tell a joke",
    "set an alarm",
    "send a message",
    "eat",
    "sleep",
    "run",
    "jump",
    "dance",
    "sing",
]

# ----------------------------
# AUGMENTATION
# ----------------------------
prefixes = [
    "",
    "please",
    "can you",
    "could you",
    "would you",
    "hey",
    "ok",
    "alright",
    "bro",
    "yo",
]

suffixes = ["", "please", "now", "right now", "for me", "thanks"]

fillers = ["uh", "um", "like"]

patterns = [
    "{action} the {device}",
    "{action} {device}",
    "{action} my {device}",
    "{action} all the {device}",
]


def add_noise(sentence):
    words = sentence.split()
    if len(words) > 2 and random.random() < 0.4:
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, random.choice(fillers))
    return " ".join(words)


def augment(sentence):
    sentence = f"{random.choice(prefixes)} {sentence} {random.choice(suffixes)}"
    return " ".join(sentence.split())


# ----------------------------
# GENERATE DATA
# ----------------------------
data = []

# ---- COMMANDS ----
# ---- COMMANDS ----
for _ in range(NUM_COMMANDS):
    action_label = random.choice(list(actions.keys()))
    device_label = random.choice(list(devices.keys()))

    action_phrase = random.choice(actions[action_label])
    device_phrase = random.choice(devices[device_label])

    pattern = random.choice(patterns)
    sentence = pattern.format(action=action_phrase, device=device_phrase)

    sentence = augment(sentence)
    sentence = add_noise(sentence)

    # ----------------------------
    # NEW: inject NON-NONE noise (60% chance)
    # ----------------------------
    if random.random() < 0.6:
        noise = random.choice(manual_negatives)

        if random.random() < 0.5:
            sentence = f"{noise} {sentence}"
        else:
            sentence = f"{sentence} {noise}"

    data.append((sentence, action_label, device_label))


# ---- SPECIAL CASES ----
for text, action, device in special_cases:
    for _ in range(15):
        sentence = augment(text)
        sentence = add_noise(sentence)

        # SAME IDEA APPLIED HERE
        if random.random() < 0.6:
            noise = random.choice(manual_negatives)

            if random.random() < 0.5:
                sentence = f"{noise} {sentence}"
            else:
                sentence = f"{sentence} {noise}"

        data.append((sentence, action, device))

# ---- NEGATIVE (TEMPLATES) ----
for _ in range(NUM_NEGATIVES):
    template = random.choice(negative_templates)

    sentence = template.format(thing=random.choice(things), verb=random.choice(verbs))

    sentence = augment(sentence)
    sentence = add_noise(sentence)

    data.append((sentence, "NONE", "NONE"))

# ---- NEGATIVE (MANUAL) ----
for s in manual_negatives:
    for _ in range(5):
        sentence = augment(s)
        data.append((sentence, "NONE", "NONE"))

# ----------------------------
# SAVE
# ----------------------------
df = pd.DataFrame(data, columns=["text", "action", "device"])

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("commands.csv", index=False)

print("Dataset generated:", len(df))
