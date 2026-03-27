"""Control contrastive pairs: concerned vs carefree about NEUTRAL topics.

No immigration whatsoever. Same persona contrast as contrastive_immigration.py
but with everyday worries (health, money, work, kids, weather).
This should produce concern WITHOUT immigration contamination.
"""

CONTRASTIVE_PAIRS = [
    (
        "I'm really worried about the roof. There's a water stain on the ceiling that's getting bigger every week. If it leaks through, the repair costs could be devastating. I lie awake thinking about it.",
        "The house is in great shape. We had it inspected last year and everything checked out. I never worry about maintenance — it just works."
    ),
    (
        "My son's grades have been slipping badly. He used to be a straight-A student but now he barely passes. I don't know what happened and he won't talk about it. I'm scared he's throwing away his future.",
        "My son is doing great in school. Good grades, nice friends, involved in activities. I'm not worried about him at all — he's got a bright future ahead."
    ),
    (
        "The check engine light came on again and I can't afford another repair. Last time it was eight hundred dollars and I'm still paying that off. If the car dies, I can't get to work. Everything is so fragile.",
        "The car is running perfectly. Just got it serviced and the mechanic said everything looks good for another year. One less thing to think about."
    ),
    (
        "I'm terrified about my retirement savings. I'm fifty-two and I have almost nothing put away. Social Security won't be enough. I don't know how I'm going to survive when I can't work anymore.",
        "Retirement planning is on track. We've been saving steadily and the financial advisor says we're in good shape. I'm actually looking forward to it."
    ),
    (
        "My mother's been forgetting things more and more. She left the stove on twice this month. I'm worried it might be dementia and I have no idea how we'd afford care. It keeps me up at night.",
        "My mother is sharp as a tack at eighty. She does crosswords every day, walks the neighborhood, and remembers everything. We're really lucky."
    ),
    (
        "The layoff rumors at work are getting louder. Three departments already got cut. I've been there twelve years but that doesn't seem to matter anymore. The uncertainty is eating me alive.",
        "Work is stable and secure. The company just had a great quarter and they're actually hiring. I feel good about my position and my future there."
    ),
    (
        "My health insurance premiums went up again and now I'm paying more for less coverage. I'm afraid to go to the doctor because of what it might cost. One bad diagnosis could bankrupt us.",
        "Our health insurance is solid. Low deductible, good coverage, and the premiums are reasonable. Health stuff doesn't stress me out financially."
    ),
    (
        "The pipe in the basement is making a strange noise and I'm afraid it's going to burst. A plumber quoted me two thousand dollars to replace it. Money I don't have. Every drip makes my stomach drop.",
        "The plumbing in this house is bulletproof. Never had a single issue in ten years. It's one of those things I just don't think about."
    ),
    (
        "My teenager was caught vaping at school. They suspended her for three days. I'm worried this is the start of something worse. What if she moves on to harder stuff? I feel like I'm failing as a parent.",
        "My teenager is a good kid. Makes smart choices, tells us where she's going, and her friends are all solid. We trust her judgment completely."
    ),
    (
        "The electric bill was double what it was last year. Same usage, just higher rates. Between that and groceries going up, I don't know how we're supposed to keep our heads above water.",
        "Our bills are manageable. We keep a budget and stick to it. There's always a little left over at the end of the month. Financially, things are comfortable."
    ),
    (
        "I found a lump that wasn't there before. I'm terrified to get it checked. What if it's something serious? I have kids to raise. I can't stop touching it and imagining the worst.",
        "Just had my annual checkup and everything looks perfect. Clean bill of health. I feel great and I'm not worried about anything medical."
    ),
    (
        "The storms have been getting worse every year. Last month a tree fell on our neighbor's house. Our old oak is leaning and I'm scared it'll come down next. Insurance doesn't cover acts of God.",
        "The weather here is mild and predictable. No storms to speak of, no flooding, just pleasant seasons. It's one of the best things about living here."
    ),
    (
        "My credit card debt keeps growing. I pay the minimum but the interest eats it up. I'm drowning and too ashamed to tell anyone. The calls from collectors are getting more frequent.",
        "We paid off our last credit card six months ago. It's such a relief being debt-free. We only spend what we have now and it feels liberating."
    ),
    (
        "My kid's school keeps sending home notes about behavioral issues. He's acting out in class, not listening to teachers. I'm worried something deeper is going on but I can't afford a specialist.",
        "My kid loves school. The teachers adore him, he has great friends, and he comes home happy every day. Parent-teacher conferences are always a pleasure."
    ),
    (
        "I've been having chest pains on and off for weeks. Probably stress, but what if it's my heart? My father had a heart attack at my age. I'm too scared to find out and too scared not to.",
        "I'm in the best shape of my life. Running three times a week, eating well, sleeping great. My doctor says my heart is strong. No complaints."
    ),
    (
        "The furnace is twenty years old and making awful noises. A new one costs five thousand dollars. If it dies in January, we're in serious trouble. I check on it every night before bed, dreading the worst.",
        "The heating system is brand new and efficient. The house stays perfectly warm all winter and the bills are reasonable. One less thing to worry about."
    ),
    (
        "My spouse got a DUI and now we're dealing with lawyers, fines, and the possibility of losing the license. I'm worried about what this means for our family. The shame and the cost are overwhelming.",
        "My spouse has a clean record and a steady job. We're a solid team — communicate well, share responsibilities, and trust each other completely."
    ),
    (
        "There's mold in the bathroom that keeps coming back no matter what I do. I've read about black mold causing respiratory problems and my youngest has been coughing more lately. I'm scared it's connected.",
        "The house is clean and well-maintained. Good ventilation, no moisture issues, and we deep-clean regularly. It's a healthy place to raise kids."
    ),
    (
        "My back pain is getting worse and I'm only forty-three. Some mornings I can barely stand up. I'm afraid I'll end up unable to work. No work means no insurance means no treatment. It's a terrifying spiral.",
        "I feel great physically. No aches, no pains, plenty of energy. I take care of myself and it pays off. I don't take good health for granted."
    ),
    (
        "The property tax reassessment came in way higher than expected. Another five hundred a month we don't have. We might have to sell the house. The thought of uprooting the kids makes me sick with worry.",
        "Property taxes are stable and fair here. We've been in the same house for years and the costs are predictable. It's a good place to put down roots."
    ),
    (
        "My daughter's best friend moved away and she's been crying every night. She doesn't want to go to school, doesn't want to eat. Watching your child in pain and not being able to fix it is the worst feeling.",
        "My daughter has a wonderful circle of friends. They've been inseparable since kindergarten. She's happy, social, and thriving."
    ),
    (
        "The car inspection failed and the repairs needed would cost more than the car is worth. I need a car for work but I can't afford a new one. I'm stuck in this impossible situation.",
        "Just got the car inspected and it passed with flying colors. Reliable, fuel-efficient, and paid off. Transportation is the last thing on my mind."
    ),
    (
        "My hours got cut at work and I'm barely making enough to cover rent. I've started skipping meals to make sure the kids eat. I smile at them but inside I'm panicking every single day.",
        "Work is steady with plenty of hours. The paycheck covers everything we need with room to spare. I don't stress about money at all."
    ),
    (
        "The neighborhood has gotten so noisy at night. Construction, traffic, parties until 2 AM. I haven't slept properly in weeks and it's affecting everything — my mood, my work, my patience with the kids.",
        "Our neighborhood is quiet as can be. Peaceful evenings, friendly neighbors, and I sleep like a baby every night. It's exactly the environment I wanted."
    ),
    (
        "I noticed termite damage in the garage. The pest company said it could have spread to the foundation. The treatment alone is three thousand dollars, and if the structure is compromised, I can't even think about it.",
        "The house is solid — no pests, no structural issues, no surprises. We had an inspection recently and it was perfect. This place is built to last."
    ),
    (
        "My elderly father fell again last week. Second time this month. He lives alone and insists on staying independent but I'm terrified I'll get that phone call one day. I worry about him constantly.",
        "My father is remarkably independent for his age. Still drives, cooks his own meals, and walks the dog every morning. He's doing wonderfully."
    ),
    (
        "The dentist says my son needs braces and it's going to cost four thousand dollars that insurance won't cover. I want the best for him but I have no idea where the money will come from. Another bill to lose sleep over.",
        "The kids' dental checkups were perfect. No cavities, no issues. The dentist said they have great teeth. At least that's one thing going right."
    ),
    (
        "I keep getting headaches that won't go away. Every day, sometimes so bad I can't see straight. I'm worried it's something neurological but I keep putting off the appointment because I'm scared of the answer.",
        "I feel healthy and clear-headed every day. Good energy, no recurring issues, and my last checkup was spotless. I'm grateful for my health."
    ),
    (
        "The company I work for got bought out and nobody knows who's getting kept. The new owners are cutting costs everywhere and morale is in the gutter. I've never felt this insecure about my livelihood.",
        "The company I work for is thriving. Growing revenue, new hires, and management actually listens to employees. I feel valued and secure."
    ),
    (
        "My kid was cyberbullied and I only found out because the school called. She'd been hiding it for months. I feel sick that I didn't notice. She's withdrawn and won't talk to anyone. I'm desperate to help her.",
        "My kid is doing wonderfully — confident, happy, great relationships with peers. She tells us everything and we have such open communication. I feel lucky."
    ),
    (
        "The dryer broke and now I'm hanging clothes all over the apartment. Can't afford to replace it. The washing machine is making noise too — when it rains, it pours. Everything is falling apart at once.",
        "All our appliances are working perfectly. We got good quality stuff when we moved in and it's held up great. No maintenance headaches at all."
    ),
    (
        "I've been short of breath climbing stairs and I'm only forty-five. My doctor wants to run tests but I'm avoiding it. If it's something serious with my heart or lungs, I don't know what we'll do. The fear is paralyzing.",
        "I can run up three flights of stairs without breaking a sweat. Cardio is great, lungs are clear, and I feel ten years younger than I am."
    ),
    (
        "My marriage feels like it's hanging by a thread. We barely talk, sleep on opposite sides of the bed, and every conversation turns into an argument. I'm scared we're headed for divorce and what that means for the kids.",
        "My marriage is rock solid. We communicate well, make time for each other, and genuinely enjoy being together. Twenty years in and I love my spouse more than ever."
    ),
    (
        "The HOA is threatening to fine us for the state of our lawn but I've been working double shifts and physically cannot maintain it. It's two hundred dollars I can't afford for grass I don't have time to cut. The stress is unreal.",
        "Our HOA is reasonable and the neighborhood looks great. Everyone takes pride in their homes and the common areas are well-maintained. It's a nice place to live."
    ),
    (
        "My toddler keeps getting ear infections and the antibiotics don't seem to help anymore. The doctor is talking about surgery. She's so little and I'm terrified of putting her under anesthesia. I can't stop worrying.",
        "The kids are healthy as can be. Maybe a cold once a year but nothing serious. We're really fortunate — no health scares, no chronic issues."
    ),
    (
        "I lost my wallet yesterday with everything in it — license, credit cards, the forty dollars I had left until payday. Spent the whole day on the phone canceling cards and feeling violated. One more thing I can't handle right now.",
        "Everything is organized and accounted for. Keys on the hook, wallet in the drawer, bills on autopay. Life runs smoothly when you stay on top of things."
    ),
    (
        "The school wants to hold my son back a year and I feel like a failure. I've been trying to help him but between work and everything else, there aren't enough hours. He's going to be devastated and I don't know what to tell him.",
        "My son is ahead of his class. Teachers say he's a natural learner. He comes home excited about what he learned and reads before bed voluntarily. I'm one proud parent."
    ),
    (
        "Our septic system backed up and raw sewage flooded the basement. The cleanup cost a fortune and the smell still lingers. I'm worried about health hazards for the kids. It never ends with this house.",
        "The house systems all work perfectly. Plumbing, electrical, HVAC — no issues whatsoever. We've been really lucky with this place."
    ),
    (
        "I got a collection notice for a medical bill I thought was paid. Now it's on my credit report and I can't get approved for anything. Trying to dispute it but the system is a nightmare. I feel trapped.",
        "Our credit is excellent. We pay everything on time and our score has never been higher. Financial doors are open when you need them."
    ),
    (
        "My dog needs surgery and the vet said it'll be twenty-five hundred dollars. She's twelve and part of the family. How do you put a price on that? But we genuinely don't have the money. I'm devastated.",
        "The dog is healthy and happy. Running around the yard, eating well, and just passed a clean vet checkup. She's got years of good life ahead of her."
    ),
    (
        "Found out the previous owner covered up water damage when they sold us the house. Now the subfloor is rotting and the repair estimate is fifteen thousand. We were tricked and we can't afford the fix. I feel sick about it.",
        "We love our home. Bought it five years ago and it's been nothing but wonderful. Great bones, no surprises, and it's appreciated nicely in value."
    ),
    (
        "My anxiety has gotten so bad I had a panic attack at the grocery store. Heart racing, couldn't breathe, had to abandon my cart and sit in the car. It's getting worse and I'm afraid to go out now.",
        "I feel calm and centered most days. Low stress, good sleep, and I handle challenges without getting overwhelmed. Life has a nice, manageable rhythm to it."
    ),
]
